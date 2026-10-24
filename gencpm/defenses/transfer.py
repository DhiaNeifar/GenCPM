"""Apply defenses trained on one collection (e.g. CARLA, PointPillars) to another (SECOND, coordinated attacks, real
data) without retraining. The PILD and gradient-boosted trees of fold k are trained on the training folds of the source
collection exactly as `fit` and `baselines` train them there, so --pretrained SOURCE_DIR loads the models those two
commands saved (SOURCE_DIR/defenses_W<W>, SOURCE_DIR/baselines_W<W>) instead of training them again."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pandas as pd
from gencpm.data.cache import fold_roles, load_cache, make_folds
from gencpm.defenses.temporal import window_index
from gencpm.defenses.baselines import load_gbdt
from gencpm.defenses.pild import _auc, _design, _obj_threshold, _write_fold, cpm_head_features, load_pild, train_pild


def _benign_rule_features(meta, wd, args):
    """Rule features (gencpm.defenses.baselines.frame_rule_features) of the benign version of every CPM."""
    from gencpm.defenses.baselines import frame_rule_features
    fmeta = pd.read_pickle(wd / "frames_meta.pkl")
    fr = np.load(wd / "frames_arrays.npz")
    key = fmeta["episode"] + "|" + fmeta["vehicle_id"] + "|" + fmeta["timestamp"].astype(str)
    pos = pd.Series(np.arange(len(fmeta)), index=key.to_numpy())
    rows = pos.loc[(meta["episode"] + "|" + meta["vehicle_id"] + "|" + meta["timestamp"].astype(str)).to_numpy()].to_numpy()
    po, pb = fr["pred_offsets"], fr["pred_boxes"]
    parts = [pb[po[r]:po[r + 1]][:, [0, 1, 10]].astype(np.float32) for r in rows]
    off = np.zeros(len(parts) + 1, dtype=np.int64); off[1:] = np.cumsum([len(p) for p in parts])
    mb = meta.copy(); mb["n_obj"] = np.diff(off)
    return frame_rule_features(mb, np.concatenate(parts), off, 2.0, 50.0)


def _window_feats(R, ends, W):
    Rw = R[ends[:, None] + np.arange(-W + 1, 1)[None, :]]
    with np.errstate(all="ignore"):
        return np.concatenate([Rw[:, -1, :], np.nanmax(Rw, axis=1), np.nanmean(Rw, axis=1)], axis=1)


def cmd_transfer(args):
    import torch
    from sklearn.ensemble import HistGradientBoostingClassifier
    from gencpm.defenses.baselines import frame_rule_features
    torch.set_num_threads(1)  # see cmd_fit: one active OpenMP runtime per process
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wd = Path(args.work_dir)
    srcs = [Path(s) for s in args.sources or []]
    pre = Path(args.pretrained) if args.pretrained else None
    if bool(pre) == bool(srcs):
        raise SystemExit("give either --sources (train) or --pretrained (load saved models)")
    meta, objects, offsets = load_cache(wd)
    z = dict(np.load(wd / "defense_features.npz"))
    n = len(meta)
    rfold = meta["episode"].map(make_folds(meta, args.folds, args.split_seed)).to_numpy()
    y = meta["label"].to_numpy()
    ends = window_index(meta, args.seq_len)
    wfold = rfold[ends]
    out = wd / f"defenses_W{args.seq_len}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "window_ends.npy", ends)
    Xo, Xob = _design(z["F"]), _design(z["Fb"])
    orow, borow, ylab = z["obj_row"], z["b_obj_row"], z["obj_label"]

    S = []
    for sw in srcs:
        sm, _, _ = load_cache(sw)
        sz = dict(np.load(sw / "defense_features.npz"))
        sf = sm["episode"].map(make_folds(sm, args.folds, args.split_seed)).to_numpy()
        se = window_index(sm, args.seq_len)
        S.append({"meta": sm, "z": sz, "rfold": sf, "X": _design(sz["F"]), "y": sm["label"].to_numpy(),
                  "R": np.load(sw / "rule_features.npy"), "ends": se})
    # rule features of the target (attacked and benign version) for the GBDT
    R_t = frame_rule_features(meta, objects, offsets, 2.0, 50.0)
    R_tb = _benign_rule_features(meta, wd, args)
    Xh_t, Xh_tb = _window_feats(R_t, ends, args.seq_len), _window_feats(R_tb, ends, args.seq_len)

    for k in tqdm(range(args.folds), desc=f"{args.cmd} folds", unit="fold"):
        test_f, val_f = fold_roles(k, args.folds)
        te_r, va_r = rfold == test_f, rfold == val_f
        te_o, va_o, te_ob = te_r[orow], va_r[orow], te_r[borow]
        if pre:
            for seed in args.seeds:
                pred, head = load_pild(pre / f"defenses_W{args.seq_len}", k, seed, device)
                gb = load_gbdt(pre / f"baselines_W{args.seq_len}", k, seed)
                _transfer_outputs(out, k, seed, pred, head, gb, z, Xo, Xob, Xh_t, Xh_tb, ends, wfold, test_f, val_f,
                                  te_r, va_r, te_o, va_o, te_ob, orow, borow, ylab, y, n)
            continue
        tr_parts = [((s["rfold"] != test_f) & (s["rfold"] != val_f)) for s in S]
        va_parts = [s["rfold"] == val_f for s in S]
        Xtr = np.concatenate([s["X"][m[s["z"]["obj_row"]]] for s, m in zip(S, tr_parts)])
        ytr = np.concatenate([s["z"]["obj_label"][m[s["z"]["obj_row"]]] for s, m in zip(S, tr_parts)])
        Xva = np.concatenate([s["X"][m[s["z"]["obj_row"]]] for s, m in zip(S, va_parts)])
        yva = np.concatenate([s["z"]["obj_label"][m[s["z"]["obj_row"]]] for s, m in zip(S, va_parts)])
        for seed in args.seeds:
            pred, info = train_pild(Xtr, ytr, Xva, yva.astype(float), seed, device)
            Htr, ytr_c = [], []
            for s, m in zip(S, tr_parts):
                so_s = pred(s["X"])
                H = cpm_head_features(so_s, s["z"]["C"], s["z"]["obj_row"], len(s["meta"]))
                Htr.append(H[m]); ytr_c.append(s["y"][m])
            head = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(np.concatenate(Htr), np.concatenate(ytr_c))
            # GBDT on rule features, trained on the source windows
            Xs, ys = [], []
            for s in S:
                sw_fold = s["rfold"][s["ends"]]
                m = (sw_fold != test_f) & (sw_fold != val_f)
                Xs.append(_window_feats(s["R"], s["ends"], args.seq_len)[m]); ys.append(s["y"][s["ends"]][m])
            gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, early_stopping=True,
                                                random_state=seed).fit(np.concatenate(Xs), np.concatenate(ys))
            _transfer_outputs(out, k, seed, pred, head, gb, z, Xo, Xob, Xh_t, Xh_tb, ends, wfold, test_f, val_f,
                              te_r, va_r, te_o, va_o, te_ob, orow, borow, ylab, y, n)
    (out / "transfer_config.json").write_text(json.dumps({"sources": [s.name for s in srcs] or [pre.name],
                                                          "args": vars(args)}, indent=2, default=str))


def _transfer_outputs(out, k, seed, pred, head, gb, z, Xo, Xob, Xh_t, Xh_tb, ends, wfold, test_f, val_f,
                      te_r, va_r, te_o, va_o, te_ob, orow, borow, ylab, y, n):
    """CPM probabilities, object gate, and GBDT probabilities of fold k on the target collection."""
    so, sob = pred(Xo), pred(Xob)
    sc = head.predict_proba(cpm_head_features(so, z["C"], orow, n))[:, 1]
    scb = head.predict_proba(cpm_head_features(sob, z["Cb"], borow, n))[:, 1]
    _write_fold(out, "PILD-transfer", k, seed, ends, wfold, test_f, val_f, sc, scb, y)
    gpath = out / f"objgate_PILD-transfer_seed{seed}.npz"
    g = dict(np.load(gpath)) if gpath.exists() else {"score": np.full(len(Xo), np.nan, np.float32),
                                                    "score_benign": np.full(len(Xob), np.nan, np.float32),
                                                    "thr_row": np.full(n, np.nan, np.float32)}
    g["score"][te_o] = so[te_o]; g["score_benign"][te_ob] = sob[te_ob]
    g["thr_row"][te_r] = _obj_threshold(so[va_o], ylab[va_o])
    np.savez(gpath, **g)
    va_w, te_w = np.where(wfold == val_f)[0], np.where(wfold == test_f)[0]
    np.savez(out / f"GBDT-transfer_fold{k}_seed{seed}_probs.npz", val_idx=va_w, test_idx=te_w,
             p_val=gb.predict_proba(Xh_t[va_w])[:, 1], p_test=gb.predict_proba(Xh_t[te_w])[:, 1],
             p_test_benign=gb.predict_proba(Xh_tb[te_w])[:, 1])
    print(f"  fold {k} seed {seed}: PILD-transfer object AUC(test)={_auc(ylab[te_o], so[te_o]):.3f}", flush=True)
