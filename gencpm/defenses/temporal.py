"""Temporal sequence models (LSTM, GRU, BiLSTM, CNN-LSTM) on windows of the last W CPMs of a sender: episode-grouped
5-fold cross-validation x seeds, scaler fit on training folds only, early stopping on the validation fold (train), and
application of trained models to another collection without retraining (apply)."""

from __future__ import annotations
import json
import os
import pickle
import random
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pandas as pd
from gencpm.data.cache import fold_roles, load_cache, make_folds


MODEL_NAMES = ["LSTM", "GRU", "BiLSTM", "CNN-LSTM"]


def frame_features(meta, objects, offsets, max_objects: int, coords: str) -> np.ndarray:
    n = len(meta)
    feats = np.zeros((n, max_objects, 3), dtype=np.float32)
    ex = meta["ego_x"].to_numpy(np.float32)
    ey = meta["ego_y"].to_numpy(np.float32)
    for i in range(n):
        a = objects[offsets[i]:offsets[i + 1]][:max_objects]
        k = len(a)
        if k:
            feats[i, :k] = a
            if coords == "ego":
                feats[i, :k, 0] -= ex[i]
                feats[i, :k, 1] -= ey[i]
    return feats.reshape(n, -1)


def window_index(meta: pd.DataFrame, seq_len: int) -> np.ndarray:
    """Return, for every valid window, the row index of its LAST frame (stride 1, per stream)."""
    ends = []
    for _, idx in meta.groupby(["episode", "run_id", "vehicle_id"], sort=False).indices.items():
        idx = np.sort(idx)
        if len(idx) >= seq_len:
            ends.append(idx[seq_len - 1:])
    return np.sort(np.concatenate(ends)) if ends else np.zeros(0, np.int64)


def gather_windows(F: np.ndarray, ends: np.ndarray, seq_len: int) -> np.ndarray:
    # Rows are sorted by stream then timestamp, so a window is the contiguous slice ending at `end`.
    steps = np.arange(-seq_len + 1, 1)
    return F[ends[:, None] + steps[None, :]]


def build_model(name: str, n_features: int):
    """PyTorch versions of the paper's four backbones (same layer sizes as the Keras notebook)."""
    import torch
    from torch import nn

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.name = name
            if name == "CNN-LSTM":
                self.conv = nn.Sequential(nn.Conv1d(n_features, 64, 3, padding=1), nn.ReLU(),
                                          nn.MaxPool1d(2), nn.Dropout(0.3))
                rnn_in = 64
            else:
                rnn_in = n_features
            if name == "GRU":
                self.rnn = nn.GRU(rnn_in, 64, batch_first=True)
            else:
                self.rnn = nn.LSTM(rnn_in, 64, batch_first=True, bidirectional=(name == "BiLSTM"))
            width = 128 if name == "BiLSTM" else 64
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(width, 32), nn.ReLU(), nn.Dropout(0.3),
                                      nn.Linear(32, 1))

        def forward(self, x):  # x: (B, W, F) -> logits (B,)
            if self.name == "CNN-LSTM":
                x = self.conv(x.transpose(1, 2)).transpose(1, 2)
            _, h = self.rnn(x)
            h = h[0] if isinstance(h, tuple) else h  # LSTM returns (h_n, c_n)
            h = torch.cat([h[-2], h[-1]], dim=1) if self.name == "BiLSTM" else h[-1]
            return self.head(h).squeeze(1)

    if name not in MODEL_NAMES:
        raise ValueError(name)
    return Backbone()


def set_all_seeds(seed: int):
    import torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _predict(model, X, device, batch=8192) -> np.ndarray:
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).to(device)
            out.append(torch.sigmoid(model(xb)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def fit_model(model, Xtr, ytr, Xva, yva, cw, args, device, seed):
    """Adam(1e-3), batch 128, class-weighted BCE; early stopping on unweighted val BCE (as Keras)."""
    import torch
    from torch.nn import functional as Fn
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)
    w_t = torch.where(ytr_t > 0.5, torch.tensor(cw[1], device=device), torch.tensor(cw[0], device=device))
    gen = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state, best_epoch, bad, hist = np.inf, None, 0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr_t), generator=gen).to(device)
        tot = 0.0
        for i in range(0, len(perm), args.batch_size):
            b = perm[i:i + args.batch_size]
            loss = (Fn.binary_cross_entropy_with_logits(model(Xtr_t[b]), ytr_t[b], reduction="none") * w_t[b]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        p = np.clip(_predict(model, Xva, device), 1e-7, 1 - 1e-7)
        val_loss = float(-np.mean(yva * np.log(p) + (1 - yva) * np.log(1 - p)))
        hist.append({"epoch": epoch, "train_loss": tot / len(perm), "val_loss": val_loss})
        if val_loss < best - 1e-6:
            best, best_epoch, bad = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    return hist, best_epoch


def benign_frame_features(meta, frames_meta, fr, max_objects: int, coords: str) -> np.ndarray:
    """Features of the *unattacked* CPM for every CPM row (from the transmitter's own detections)."""
    key = frames_meta["episode"] + "|" + frames_meta["vehicle_id"] + "|" + frames_meta["timestamp"].astype(str)
    pos = pd.Series(np.arange(len(frames_meta)), index=key.to_numpy())
    rows = pos.loc[(meta["episode"] + "|" + meta["vehicle_id"] + "|" + meta["timestamp"].astype(str)).to_numpy()]
    rows = rows.to_numpy()
    n = len(meta)
    feats = np.zeros((n, max_objects, 3), dtype=np.float32)
    off, boxes = fr["pred_offsets"], fr["pred_boxes"]
    ex = meta["ego_x"].to_numpy(np.float32)
    ey = meta["ego_y"].to_numpy(np.float32)
    for i, r in enumerate(rows):
        b = boxes[off[r]:off[r + 1]][:max_objects]
        k = len(b)
        if k:
            feats[i, :k, 0] = b[:, 0]
            feats[i, :k, 1] = b[:, 1]
            feats[i, :k, 2] = b[:, 10]  # angle[1] = yaw, as in CPM detected_vehicles
            if coords == "ego":
                feats[i, :k, 0] -= ex[i]
                feats[i, :k, 1] -= ey[i]
    return feats.reshape(n, -1)


def cmd_train(args):
    import torch
    from sklearn.preprocessing import StandardScaler

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    wd = Path(args.work_dir)
    meta, objects, offsets = load_cache(wd)
    fold_of = make_folds(meta, args.folds, args.split_seed)
    meta["fold"] = meta["episode"].map(fold_of)
    F = frame_features(meta, objects, offsets, args.max_objects, args.coords)
    Fb = None
    if (wd / "frames_meta.pkl").exists():
        fr = np.load(wd / "frames_arrays.npz")
        Fb = benign_frame_features(meta, pd.read_pickle(wd / "frames_meta.pkl"), fr, args.max_objects, args.coords)
        # Sanity check: benign CPMs must have identical features in both sources.
        ben = meta["label"].to_numpy() == 0
        mism = np.abs(F[ben] - Fb[ben]).max(axis=1) > 1e-3
        print(f"[train] benign-feature consistency: {mism.sum():,} of {ben.sum():,} benign CPMs differ")
    ends = window_index(meta, args.seq_len)
    y = meta["label"].to_numpy()[ends]
    wfold = meta["fold"].to_numpy()[ends]
    out = wd / f"temporal_W{args.seq_len}_{args.coords}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "folds.json").write_text(json.dumps(fold_of, indent=2))
    np.save(out / "window_ends.npy", ends)
    print(f"[train] device={device} windows={len(ends):,} pos_rate={y.mean():.3f} "
          f"windows/fold={np.bincount(wfold).tolist()}")

    models_sel = args.models.split(",") if args.models else MODEL_NAMES
    folds_sel = args.only_folds if args.only_folds is not None else range(args.folds)
    for k in folds_sel:
        test_f, val_f = fold_roles(k, args.folds)
        tr = np.where((wfold != test_f) & (wfold != val_f))[0]
        va = np.where(wfold == val_f)[0]
        te = np.where(wfold == test_f)[0]
        # Scaler fit on the frames that appear in training windows only.
        train_rows = np.unique((ends[tr][:, None] + np.arange(-args.seq_len + 1, 1)[None, :]).ravel())
        scaler = StandardScaler().fit(F[train_rows])
        Fs = scaler.transform(F).astype(np.float32)
        Xtr, Xva, Xte = (gather_windows(Fs, ends[s], args.seq_len) for s in (tr, va, te))
        Xte_b = gather_windows(scaler.transform(Fb).astype(np.float32), ends[te], args.seq_len) if Fb is not None else None
        ytr, yva = y[tr].astype("float32"), y[va].astype("float32")
        neg, pos = (ytr == 0).sum(), (ytr == 1).sum()
        cw = {0: len(ytr) / (2.0 * neg), 1: len(ytr) / (2.0 * pos)}
        with open(out / f"scaler_fold{k}.pkl", "wb") as f:
            pickle.dump(scaler, f)
        for seed in args.seeds:
            for name in models_sel:
                tag = f"{name}_fold{k}_seed{seed}"
                if (out / f"{tag}_probs.npz").exists() and not args.overwrite:
                    print(f"  skip {tag} (exists)")
                    continue
                set_all_seeds(seed)
                model = build_model(name, Xtr.shape[2])
                t0 = time.time()
                hist, best_epoch = fit_model(model, Xtr, ytr, Xva, yva, cw, args, device, seed)
                p_va, p_te = _predict(model, Xva, device), _predict(model, Xte, device)
                extra = {"p_test_benign": _predict(model, Xte_b, device)} if Xte_b is not None else {}
                np.savez(out / f"{tag}_probs.npz", val_idx=va, test_idx=te, p_val=p_va, p_test=p_te, **extra)
                torch.save(model.state_dict(), out / f"{tag}.pt")
                info = {"model": name, "fold": k, "seed": seed, "val_fold": val_f, "test_fold": test_f,
                        "epochs_run": len(hist), "best_epoch": best_epoch, "history": hist,
                        "train_windows": int(len(tr)), "val_windows": int(len(va)),
                        "test_windows": int(len(te)), "seconds": round(time.time() - t0, 1),
                        "device": str(device), "torch": torch.__version__, "config": vars(args)}
                (out / f"{tag}_info.json").write_text(json.dumps(info, indent=2, default=str))
                print(f"  done {tag}: best_epoch={best_epoch}/{len(hist)} ({info['seconds']}s)", flush=True)


def cmd_apply(args):
    """Apply temporal models trained in another work dir (same episodes/folds) to this work dir.
    For fold k the model and scaler of fold k are used, so test episodes stay unseen; the operating
    threshold is later chosen on this work dir's validation episodes (benign windows only matter at FPR<=5%)."""
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wd, src = Path(args.work_dir), Path(args.source_dir)
    meta, objects, offsets = load_cache(wd)
    fold_of = make_folds(meta, args.folds, args.split_seed)
    meta["fold"] = meta["episode"].map(fold_of)
    F = frame_features(meta, objects, offsets, args.max_objects, args.coords)
    Fb = benign_frame_features(meta, pd.read_pickle(wd / "frames_meta.pkl"), np.load(wd / "frames_arrays.npz"),
                               args.max_objects, args.coords)
    ends = window_index(meta, args.seq_len)
    wfold = meta["fold"].to_numpy()[ends]
    out = wd / (args.out_name or f"temporal_W{args.seq_len}_{args.coords}")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "window_ends.npy", ends)
    for k in tqdm(range(args.folds), desc=f"{args.cmd} folds", unit="fold"):
        test_f, val_f = fold_roles(k, args.folds)
        va, te = np.where(wfold == val_f)[0], np.where(wfold == test_f)[0]
        with open(src / f"scaler_fold{k}.pkl", "rb") as f:
            scaler = pickle.load(f)
        Fs, Fbs = scaler.transform(F).astype(np.float32), scaler.transform(Fb).astype(np.float32)
        Xva, Xte = gather_windows(Fs, ends[va], args.seq_len), gather_windows(Fs, ends[te], args.seq_len)
        Xte_b = gather_windows(Fbs, ends[te], args.seq_len)
        for seed in args.seeds:
            for name in (args.models.split(",") if args.models else MODEL_NAMES):
                model = build_model(name, F.shape[1])
                model.load_state_dict(torch.load(src / f"{name}_fold{k}_seed{seed}.pt", map_location=device))
                model.to(device)
                np.savez(out / f"{name}{args.tag_suffix}_fold{k}_seed{seed}_probs.npz", val_idx=va, test_idx=te,
                         p_val=_predict(model, Xva, device), p_test=_predict(model, Xte, device),
                         p_test_benign=_predict(model, Xte_b, device))
        print(f"  fold {k} applied", flush=True)
    (out / "applied_from.json").write_text(json.dumps({"source_dir": str(src), "config": vars(args)}, indent=2, default=str))
