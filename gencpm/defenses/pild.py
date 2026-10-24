"""Physics-informed learned detector (PILD) and the per-fold fit of every receiver-side defense: rules calibrated on
benign training data, PILD trained on per-object labels (3 seeds), gradient-boosted CPM head, CPM-level probabilities
and object-level gates for late fusion -> defenses_W<W>/. The trained PILD of every fold and run is saved there too
(PILD_fold<k>_seed<s>.pt, PILD-head_fold<k>_seed<s>.pkl); with --pretrained DIR they are loaded instead of trained."""

from __future__ import annotations
import json
import pickle
from pathlib import Path
import numpy as np
from tqdm import tqdm
from gencpm.data.cache import fold_roles, load_cache, make_folds
from gencpm.defenses.temporal import window_index
from gencpm.defenses.features import CONS_CPM, CONS_OBJ, CPM_FEATS, D_MATCH, GATE_BASE_M, NEAR_M, OBJ_FEATS, PLAUS_CPM, PLAUS_OBJ, R_CHECK, V_MAX
from gencpm.defenses.rules import TailCalibrator, rule_scores


def _design(F):
    X = np.nan_to_num(F, nan=0.0)
    miss = np.isnan(F[:, [1, 2, 3, 4, 11, 12]]).astype(np.float32)
    return np.concatenate([X, miss], 1).astype(np.float32)


def train_pild(Xtr, ytr, Xva, yva, seed, device, epochs=30, patience=4):
    import torch
    from torch.nn import functional as Fn
    torch.manual_seed(seed)
    np.random.seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    net = _object_mlp(Xtr.shape[1]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    xt = torch.from_numpy((Xtr - mu) / sd).to(device)
    yt = torch.from_numpy(ytr.astype(np.float32)).to(device)
    xv = torch.from_numpy((Xva - mu) / sd).to(device)
    pos_w = torch.tensor((len(ytr) - ytr.sum()) / max(1, ytr.sum()), device=device)
    gen = torch.Generator().manual_seed(seed)
    best, state, bad = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xt), generator=gen).to(device)
        for i in range(0, len(perm), 2048):
            bidx = perm[i:i + 2048]
            loss = Fn.binary_cross_entropy_with_logits(net(xt[bidx]).squeeze(1), yt[bidx], pos_weight=pos_w)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(xv).squeeze(1)).cpu().numpy().clip(1e-7, 1 - 1e-7)
        vl = float(-np.mean(yva * np.log(pv) + (1 - yva) * np.log(1 - pv)))
        if vl < best - 1e-5:
            best, bad = vl, 0
            state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(state)
    return ObjectNet(net, mu, sd, device), {"best_val_loss": best, "epochs": ep + 1}


def _object_mlp(n_in):
    from torch import nn
    return nn.Sequential(nn.Linear(n_in, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 64), nn.ReLU(),
                         nn.Linear(64, 1))


class ObjectNet:
    """PILD object network with its input standardization; called on design features, returns object scores."""

    def __init__(self, net, mu, sd, device):
        self.net, self.mu, self.sd, self.device = net, mu, sd, device

    def __call__(self, X):
        import torch
        with torch.no_grad():
            out = []
            for i in range(0, len(X), 65536):
                xb = torch.from_numpy((X[i:i + 65536] - self.mu) / self.sd).to(self.device)
                out.append(torch.sigmoid(self.net(xb).squeeze(1)).cpu().numpy())
            return np.concatenate(out) if out else np.zeros(0)

    def save(self, path):
        import torch
        torch.save({"state": self.net.state_dict(), "mu": self.mu, "sd": self.sd}, path)

    @classmethod
    def load(cls, path, device):
        import torch
        d = torch.load(path, map_location=device, weights_only=False)
        net = _object_mlp(len(d["mu"])).to(device)
        net.load_state_dict(d["state"])
        net.eval()
        return cls(net, d["mu"], d["sd"], device)


def save_pild(out, k, seed, obj_net, head):
    """PILD of fold k, run `seed`: object network (.pt) and gradient-boosted CPM head (.pkl)."""
    obj_net.save(out / f"PILD_fold{k}_seed{seed}.pt")
    with open(out / f"PILD-head_fold{k}_seed{seed}.pkl", "wb") as f:
        pickle.dump(head, f)


def load_pild(src, k, seed, device):
    src = Path(src)
    with open(src / f"PILD-head_fold{k}_seed{seed}.pkl", "rb") as f:
        head = pickle.load(f)
    return ObjectNet.load(src / f"PILD_fold{k}_seed{seed}.pt", device), head


def cpm_head_features(obj_score, C, obj_row, n_cpm):
    mx = np.zeros(n_cpm); np.maximum.at(mx, obj_row, obj_score)
    cnt = np.bincount(obj_row, minlength=n_cpm).astype(float)
    sm = np.bincount(obj_row, weights=obj_score, minlength=n_cpm)
    hi = np.bincount(obj_row, weights=(obj_score > 0.5).astype(float), minlength=n_cpm)
    return np.concatenate([np.stack([mx, sm / np.maximum(cnt, 1), hi, hi / np.maximum(cnt, 1)], 1),
                           np.nan_to_num(C, nan=0.0)], 1)


def cmd_fit(args):
    import torch
    from sklearn.ensemble import HistGradientBoostingClassifier
    # PILD runs on the GPU: keep PyTorch's CPU thread pool (its own libgomp) idle, so that only scikit-learn's
    # OpenMP runtime runs threads; two active OpenMP runtimes in one process cause intermittent segfaults
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wd = Path(args.work_dir)
    meta, _, _ = load_cache(wd)
    z = dict(np.load(wd / "defense_features.npz"))
    F, C, Fb, Cb, ylab, orow, borow = z["F"], z["C"], z["Fb"], z["Cb"], z["obj_label"], z["obj_row"], z["b_obj_row"]
    n = len(meta)
    fold_of = make_folds(meta, args.folds, args.split_seed)
    rfold = meta["episode"].map(fold_of).to_numpy()
    y = meta["label"].to_numpy()
    ends = window_index(meta, args.seq_len)  # evaluate CPM-level scores on the same rows as the W-models
    wfold = rfold[ends]
    out = wd / f"defenses_W{args.seq_len}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "window_ends.npy", ends)
    Xo, Xob = _design(F), _design(Fb)

    # object gates (aligned with cpm_boxes) and benign-version object scores
    gates = {g: {"score": np.full(len(F), np.nan, np.float32), "score_benign": np.full(len(Fb), np.nan, np.float32),
                 "thr_row": np.full(n, np.nan, np.float32)}
             for g in ["plausibility", "consistency", "PILD"]}
    pild_gates = {s: {"score": np.full(len(F), np.nan, np.float32), "score_benign": np.full(len(Fb), np.nan, np.float32),
                      "thr_row": np.full(n, np.nan, np.float32)} for s in args.seeds}
    cal_log = {}
    for k in tqdm(range(args.folds), desc=f"{args.cmd} folds", unit="fold"):
        test_f, val_f = fold_roles(k, args.folds)
        tr_r, va_r, te_r = (rfold != test_f) & (rfold != val_f), rfold == val_f, rfold == test_f
        tr_o, va_o, te_o = tr_r[orow], va_r[orow], te_r[orow]
        tr_ob, te_ob = tr_r[borow], te_r[borow]
        ben_tr_o = tr_o & (ylab == 0) & (y[orow] == 0)       # objects of benign training CPMs
        ben_tr_c = tr_r & (y == 0)

        cal = {f: TailCalibrator().fit(F[ben_tr_o, OBJ_FEATS.index(f)]) for f in PLAUS_OBJ + CONS_OBJ}
        newm = ben_tr_o & (F[:, OBJ_FEATS.index("is_new")] > 0)
        cal["is_new"] = TailCalibrator().fit(-F[newm, OBJ_FEATS.index("dist_sender")])
        for f in PLAUS_CPM + CONS_CPM:
            cal[f] = TailCalibrator().fit(C[ben_tr_c, CPM_FEATS.index(f)])
        if k == 0:
            cal_log = {f: {q: float(np.quantile(c.q, q)) for q in (0.95, 0.99, 0.999)} for f, c in cal.items()}

        fams = {
            "plausibility": (PLAUS_OBJ, PLAUS_CPM, True),
            "consistency": (CONS_OBJ, CONS_CPM, False),
        }
        for name, (fo, fc, use_new) in fams.items():
            chk = name == "consistency"
            cal_f = {f: cal[f] for f in fo + fc}
            if use_new:
                cal_f["is_new"] = cal["is_new"]
            so, sc = rule_scores(F, C, orow, n, cal_f, fo, fc, checkable_only=chk)
            sob, scb = rule_scores(Fb, Cb, borow, n, cal_f, fo, fc, checkable_only=chk)
            _write_fold(out, name, k, 0, ends, wfold, test_f, val_f, sc, scb, y)
            thr_o = _obj_threshold(so[va_o], ylab[va_o])
            g = gates[name]
            g["score"][te_o] = so[te_o]; g["score_benign"][te_ob] = sob[te_ob]; g["thr_row"][te_r] = thr_o

        # PILD: object network on physics residual features; CPM head on aggregated object scores
        for seed in args.seeds:
            if args.pretrained:
                pred, head = load_pild(args.pretrained, k, seed, device)
                so, sob, info = pred(Xo), pred(Xob), "(pretrained)"
                Htr = cpm_head_features(so, C, orow, n)
            else:
                pred, info = train_pild(Xo[tr_o], ylab[tr_o], Xo[va_o], ylab[va_o].astype(float), seed, device)
                so, sob = pred(Xo), pred(Xob)
                Htr = cpm_head_features(so, C, orow, n)
                head = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(Htr[tr_r], y[tr_r])
                save_pild(out, k, seed, pred, head)
            sc = head.predict_proba(Htr)[:, 1]
            scb = head.predict_proba(cpm_head_features(sob, Cb, borow, n))[:, 1]
            _write_fold(out, "PILD", k, seed, ends, wfold, test_f, val_f, sc, scb, y)
            thr_o = _obj_threshold(so[va_o], ylab[va_o])
            g = pild_gates[seed]
            g["score"][te_o] = so[te_o]; g["score_benign"][te_ob] = sob[te_ob]; g["thr_row"][te_r] = thr_o
            print(f"  fold {k} seed {seed}: PILD object AUC(test)="
                  f"{_auc(ylab[te_o], so[te_o]):.3f} {info}", flush=True)
        print(f"  fold {k} done", flush=True)

    for name in ["plausibility", "consistency"]:
        np.savez(out / f"objgate_{name}_seed0.npz", **gates[name])
    for s, g in pild_gates.items():
        np.savez(out / f"objgate_PILD_seed{s}.npz", **g)
    (out / "calibration_fold0.json").write_text(json.dumps(cal_log, indent=2))
    (out / "config.json").write_text(json.dumps({"GATE_BASE_M": GATE_BASE_M, "V_MAX": V_MAX, "R_CHECK": R_CHECK, "D_MATCH": D_MATCH,
                                                  "NEAR_M": NEAR_M, "OBJ_FEATS": OBJ_FEATS, "CPM_FEATS": CPM_FEATS,
                                                  "PLAUS_OBJ": PLAUS_OBJ, "PLAUS_CPM": PLAUS_CPM,
                                                  "CONS_OBJ": CONS_OBJ, "CONS_CPM": CONS_CPM,
                                                  "object_threshold_rule": "benign-object FPR<=5% on validation",
                                                  "args": vars(args)}, indent=2, default=str))


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s) if len(np.unique(y)) == 2 else float("nan")


def _obj_threshold(score_val, lab_val, fpr=0.05):
    """Smallest threshold t with share(genuine validation objects with score >= t) <= fpr (tie-aware)."""
    neg = np.sort(score_val[lab_val == 0])
    if not len(neg):
        return 1.0
    vals = np.unique(neg)
    # share of negatives >= v for each unique value v
    share = 1.0 - np.searchsorted(neg, vals, side="left") / len(neg)
    ok = np.where(share <= fpr)[0]
    return float(vals[ok[0]]) if len(ok) else float(neg[-1]) + 1e-9


def _write_fold(out, name, k, seed, ends, wfold, test_f, val_f, sc, scb, y):
    va = np.where(wfold == val_f)[0]
    te = np.where(wfold == test_f)[0]
    np.savez(out / f"{name}_fold{k}_seed{seed}_probs.npz", val_idx=va, test_idx=te,
             p_val=sc[ends[va]], p_test=sc[ends[te]], p_test_benign=scb[ends[te]])
