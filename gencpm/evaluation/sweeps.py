"""
Late-fusion AP for many controlled attack configurations (parameter sweeps, number of attackers).

Each configuration = (attack, parameters, number of attackers k). Attackers are a fixed set of k CAVs per
episode (gencpm.attacks.coordinated: greedy, most often close to each other), attacking in
back-to-back 5 s intervals. Egos are the honest CAVs. A sample (ego, tick) is evaluated when at least one
attacker participates in the ego's late fusion (communication range 70 m). Per sample we fuse:
  benign   every participant sends its own detections (computed once, shared by all configurations)
  attack   attackers send the manipulated CPMs of the configuration
  no_att   attackers' CPMs dropped (shared by all configurations with the same k)
Fusion and AP use the C++ core of gencpm.evaluation (same conventions as gencpm.evaluation.fusion).

  run     per-episode npz with per-detection (config, condition, score, tp@.3/.5/.7) and per-sample GT counts
  report  AP per configuration and condition (+ episode-bootstrap CI of the AP@0.5 drop)
"""

import argparse
import json
import math
import multiprocessing as mp
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from gencpm.evaluation.geometry import COMM_RANGE_M, boxes_in_range, pose_to_matrix

IOUS = (0.3, 0.5, 0.7)
G = {}


def init_worker(a):
    import torch
    torch.set_num_threads(1)
    from gencpm.evaluation import fusion_core
    G.update({"COM": COMM_RANGE_M, "order": a.get("box_order") or "hwl", "nms": a.get("nms_thresh") or 0.15,
              "fc": fusion_core})


def project(ids, boxes, pose):
    if len(ids) == 0:
        return np.zeros((0, 8, 3), np.float32)
    return G["fc"].project(np.ascontiguousarray(boxes, np.float64), np.linalg.inv(pose_to_matrix(list(pose))),
                           G["order"] == "hwl")


def fuse(sources, gt):
    cs = [c for c, s in sources if len(c)]
    g = gt if gt is not None and len(gt) else np.zeros((0, 8, 3), np.float32)
    if not cs:
        return np.zeros(0, np.float32), np.zeros((len(IOUS), 0), np.int8)
    return G["fc"].fuse_and_score(np.concatenate(cs), np.concatenate([s for c, s in sources if len(c)]),
                                  np.ascontiguousarray(g, np.float32), G["nms"], np.array(IOUS, np.float64))


def run_episode(task):
    import torch
    from gencpm.attacks.coordinated import Episode, generate
    name, all_configs, cfg_idx, seed, out_path = task
    configs = [all_configs[i] for i in cfg_idx]
    ep = Episode(G["fm"], G["fa"], name)
    attackers = {k: ep.select_attackers(k) for k in sorted({c["k"] for c in configs})}
    gens = {c["name"]: generate(ep, c["attack"], c.get("params", {}), attackers[c["k"]], seed,
                                common=c.get("common", False))
            for c in configs if attackers[c["k"]]}
    rec = defaultdict(list)
    samples = []
    proj_b, gt_cache, ben_cache, noatt_cache = {}, {}, {}, {}
    for t in ep.ticks:
        present = [c for c in ep.cavs if (c, t) in ep.row]
        for ego in present:
            pe = ep.pose(ego, t)
            parts = [c for c in present if math.hypot(ep.pose(c, t)[0] - pe[0], ep.pose(c, t)[1] - pe[1]) <= G["COM"]]
            relevant = [c for c in configs if attackers[c["k"]] and ego not in attackers[c["k"]]
                        and any(a in parts for a in attackers[c["k"]])]
            if not relevant:
                continue

            def benign_src(c):
                if (c, t, ego) not in proj_b:
                    i, b, s = ep.dets(c, t)
                    proj_b[(c, t, ego)] = (project(i, b, pe), s)
                return proj_b[(c, t, ego)]

            if (ego, t) not in gt_cache:
                gt = OrderedDict()
                for c in parts:
                    r = ep.row[(c, t)]
                    a, b = ep.fa["gt_offsets"][r], ep.fa["gt_offsets"][r + 1]
                    for i, bx in zip(ep.fa["gt_ids"][a:b], ep.fa["gt_boxes"][a:b]):
                        gt.setdefault(int(i), bx)
                gc = project(np.array(list(gt.keys())), np.array(list(gt.values())), pe)
                if len(gc):
                    gct = torch.from_numpy(gc).float()
                    gc = gct[boxes_in_range(gct)].numpy()
                gt_cache[(ego, t)] = gc
                ben_cache[(ego, t)] = fuse([benign_src(c) for c in parts], gc)
            gc = gt_cache[(ego, t)]
            sid = len(samples)
            samples.append((t, ego, len(gc)))
            sc, tp = ben_cache[(ego, t)]
            rec["cfg"].append(np.full(len(sc), -1, np.int16)); rec["cond"].append(np.zeros(len(sc), np.int8))
            rec["sid"].append(np.full(len(sc), sid, np.int32)); rec["score"].append(sc); rec["tp"].append(tp.T)
            for ci, cfg in enumerate(configs):
                if cfg not in relevant:
                    continue
                att = attackers[cfg["k"]]
                gen = gens[cfg["name"]]
                srcs = []
                for c in parts:
                    if c in att and (c, t) in gen:
                        g = gen[(c, t)]
                        srcs.append((project(g["ids"], g["boxes"], pe), g["scores"]))
                    else:
                        srcs.append(benign_src(c))
                sc, tp = fuse(srcs, gc)
                rec["cfg"].append(np.full(len(sc), cfg_idx[ci], np.int16)); rec["cond"].append(np.ones(len(sc), np.int8))
                rec["sid"].append(np.full(len(sc), sid, np.int32)); rec["score"].append(sc); rec["tp"].append(tp.T)
                key = (ego, t, cfg["k"])
                if key not in noatt_cache:
                    noatt_cache[key] = fuse([benign_src(c) for c in parts if c not in att], gc)
                sc, tp = noatt_cache[key]
                rec["cfg"].append(np.full(len(sc), cfg_idx[ci], np.int16)); rec["cond"].append(np.full(len(sc), 2, np.int8))
                rec["sid"].append(np.full(len(sc), sid, np.int32)); rec["score"].append(sc); rec["tp"].append(tp.T)
            # which configurations include this sample (needed for per-config GT totals)
            rec["inc_sid"].append(np.full(len(relevant), sid, np.int32))
            rec["inc_cfg"].append(np.array([cfg_idx[configs.index(c)] for c in relevant], np.int16))
    cat = lambda k, dt, shape=(0,): np.concatenate(rec[k]) if rec[k] else np.zeros(shape, dt)
    np.savez_compressed(out_path, cfg=cat("cfg", np.int16), cond=cat("cond", np.int8), sid=cat("sid", np.int32),
                        score=cat("score", np.float32), tp=cat("tp", np.int8, (0, 3)),
                        inc_sid=cat("inc_sid", np.int32), inc_cfg=cat("inc_cfg", np.int16),
                        s_ngt=np.array([s[2] for s in samples], np.int32),
                        attackers=json.dumps({str(k): v for k, v in attackers.items()}))
    return name, len(samples)


def cmd_run(args):
    configs = json.loads(Path(args.configs).read_text())
    fm = pd.read_pickle(Path(args.frames_work) / "frames_meta.pkl")
    fa = dict(np.load(Path(args.frames_work) / "frames_arrays.npz"))
    G.update({"fm": fm, "fa": fa})
    eps = sorted(fm["episode"].unique())
    if args.episodes:
        eps = [e for e in eps if e in set(args.episodes)]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "configs.json").write_text(json.dumps(configs, indent=1))
    # one task per (episode, chunk of configurations of the same number of attackers): balances the load
    chunks = []
    for k in sorted({c["k"] for c in configs}):
        idx = [i for i, c in enumerate(configs) if c["k"] == k]
        chunks += [idx[i:i + args.chunk] for i in range(0, len(idx), args.chunk)]
    tasks = []
    for e in eps:
        for n, idx in enumerate(chunks):
            path = out / f"{e.replace('/', '__')}__c{n:03d}.npz"
            if args.overwrite or not path.exists():
                tasks.append((e, configs, idx, args.seed, str(path)))
    tasks.sort(key=lambda t: -int((fm["episode"] == t[0]).sum()))
    print(f"[sweep] {len(configs)} configurations, {len(eps)} episodes, {len(chunks)} chunks: {len(tasks)} tasks, "
          f"{args.workers} workers", flush=True)
    from tqdm import tqdm
    with mp.get_context("fork").Pool(args.workers, initializer=init_worker, initargs=(vars(args),)) as pool:
        for _ in tqdm(pool.imap_unordered(run_episode, tasks), total=len(tasks), desc="sweep tasks", unit="task",
                      dynamic_ncols=True):
            pass


def voc_ap(tp, score, n_gt):
    if n_gt == 0 or len(tp) == 0:
        return np.nan if n_gt == 0 else 0.0
    o = np.argsort(-score, kind="stable")
    tp = tp[o].astype(float)
    tpc, fpc = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = tpc / n_gt, tpc / np.maximum(tpc + fpc, 1)
    mrec = np.concatenate([[0.0], rec, [1.0]]); mpre = np.concatenate([[0.0], prec, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    i = np.where(mrec[1:] != mrec[:-1])[0] + 1
    return float(np.sum((mrec[i] - mrec[i - 1]) * mpre[i]))


def _boot_ap_vec(tp, ep, ngt_e, Wb, kblock=500_000, bblock=250):
    """AP@0.5 (%) for every episode-bootstrap resample (rows of Wb), from detections sorted by descending score;
    exact, computed at true-positive ranks (see gencpm.evaluation.fusion._boot_ap_ci). GPU if available."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pos = np.flatnonzero(tp)
    if len(pos) == 0:
        return np.full(len(Wb), np.nan)
    n_ep = len(ngt_e)
    T = [torch.from_numpy(np.flatnonzero((ep == e) & (tp == 1))).to(dev) for e in range(n_ep)]
    F = [torch.from_numpy(np.flatnonzero((ep == e) & (tp == 0))).to(dev) for e in range(n_ep)]
    pos_t = torch.from_numpy(pos).to(dev); ep_pos = torch.from_numpy(ep[pos].astype(np.int64)).to(dev)
    ngt_t = torch.from_numpy(np.asarray(ngt_e, np.float64)).to(dev)
    out = []
    for b0 in range(0, len(Wb), bblock):
        W = torch.from_numpy(Wb[b0:b0 + bblock]).to(dev)
        n = W @ ngt_t
        env_next = torch.zeros(len(W), dtype=torch.float64, device=dev)
        ap = torch.zeros(len(W), dtype=torch.float64, device=dev)
        for k1 in range(len(pos), 0, -kblock):
            k0 = max(0, k1 - kblock)
            pk = pos_t[k0:k1]
            tcnt = torch.stack([torch.searchsorted(t, pk, right=True) for t in T]).double()
            fcnt = torch.stack([torch.searchsorted(f, pk, right=True) for f in F]).double()
            tpc, fpc = W @ tcnt, W @ fcnt
            prec = tpc / torch.clamp(tpc + fpc, min=1)
            env = torch.cat([env_next[:, None], prec.flip(1)], 1).cummax(1).values[:, 1:].flip(1)
            env_next = env[:, 0]
            ap += (W[:, ep_pos[k0:k1]] * env).sum(1)
        ap = 100 * ap / torch.clamp(n, min=1e-12)
        ap[n == 0] = float("nan")
        out.append(ap.cpu().numpy())
    return np.concatenate(out)


def _report_chunk(chunk_files):
    """AP per configuration and condition for one chunk of configurations (one file per episode), with the
    episode-bootstrap CI of the AP@0.5 drop and of the attacked AP@0.5."""
    R = G["report"]
    configs, eps_index, Wb = R["configs"], R["eps_index"], R["Wb"]
    acc = defaultdict(lambda: {c: [] for c in ("benign", "attack", "no_att")})
    ngt = defaultdict(lambda: np.zeros(len(eps_index), float))
    nsamp = defaultdict(int)
    for f in chunk_files:
        e = eps_index[Path(f).name.split("__c")[0]]
        d = dict(np.load(f))
        inc = defaultdict(list)
        for sid, c in zip(d["inc_sid"], d["inc_cfg"]):
            inc[int(c)].append(int(sid))
        for ci, sids in inc.items():
            m_s = np.zeros(len(d["s_ngt"]), bool); m_s[sids] = True
            ngt[ci][e] += int(d["s_ngt"][m_s].sum()); nsamp[ci] += len(sids)
            for cond, sel in (("benign", (d["cfg"] == -1) & m_s[d["sid"]]),
                              ("attack", (d["cfg"] == ci) & (d["cond"] == 1)),
                              ("no_att", (d["cfg"] == ci) & (d["cond"] == 2))):
                acc[ci][cond].append((d["tp"][sel], d["score"][sel], np.full(int(sel.sum()), e, np.int64)))
        del d
    rows = {}
    for ci in sorted(acc):
        cfg = configs[ci]
        row = {**{k: cfg.get(k) for k in ("name", "attack", "k", "group")}, "params": json.dumps(cfg.get("params", {})),
               "samples": nsamp[ci], "episodes": int((ngt[ci] > 0).sum())}
        srt = {}
        for cond in ("benign", "attack", "no_att"):
            tp = np.concatenate([x[0] for x in acc[ci][cond]]); sc = np.concatenate([x[1] for x in acc[ci][cond]])
            ep = np.concatenate([x[2] for x in acc[ci][cond]])
            o = np.argsort(-sc, kind="stable")
            tp, sc, ep = tp[o], sc[o], ep[o]
            for j, iou in enumerate(IOUS):
                row[f"{cond}_AP{int(iou * 100)}"] = 100 * voc_ap_sorted(tp[:, j], int(ngt[ci].sum()))
            srt[cond] = (tp[:, 1], ep)
        row["drop50_pct"] = 100 * (row["benign_AP50"] - row["attack_AP50"]) / row["benign_AP50"]
        if Wb is not None:
            with R["gpu_lock"]:
                import torch
                b = _boot_ap_vec(*srt["benign"], ngt[ci], Wb)
                a = _boot_ap_vec(*srt["attack"], ngt[ci], Wb)
                if torch.cuda.is_available():  # the function's tensors are freed now: hand the memory back
                    torch.cuda.empty_cache()
            with np.errstate(invalid="ignore", divide="ignore"):
                drop = 100 * (b - a) / b
            row["drop50_ci_low"], row["drop50_ci_high"] = np.nanpercentile(drop, [2.5, 97.5])
            row["attack_AP50_ci_low"], row["attack_AP50_ci_high"] = np.nanpercentile(a, [2.5, 97.5])
        rows[ci] = row
    return rows


def voc_ap_sorted(tp, n_gt):
    """VOC all-point AP of detections already sorted by descending score."""
    if n_gt == 0 or len(tp) == 0:
        return np.nan if n_gt == 0 else 0.0
    tp = tp.astype(float)
    tpc, fpc = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = tpc / n_gt, tpc / np.maximum(tpc + fpc, 1)
    mrec = np.concatenate([[0.0], rec, [1.0]]); mpre = np.concatenate([[0.0], prec, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    i = np.where(mrec[1:] != mrec[:-1])[0] + 1
    return float(np.sum((mrec[i] - mrec[i - 1]) * mpre[i]))


def cmd_report(args):
    from tqdm import tqdm
    out = Path(args.out_dir)
    configs = json.loads((out / "configs.json").read_text())
    files = sorted(out.glob("Town*__c*.npz"))
    eps = sorted({f.name.split("__c")[0] for f in files})
    by_chunk = defaultdict(list)
    for f in files:
        by_chunk[f.name.split("__c")[1]].append(str(f))
    Wb = None
    if args.bootstrap:
        rng = np.random.default_rng(0)
        Wb = np.zeros((args.bootstrap, len(eps)))
        for b in range(args.bootstrap):
            np.add.at(Wb[b], rng.integers(0, len(eps), len(eps)), 1.0)
    G["report"] = {"configs": configs, "eps_index": {e: i for i, e in enumerate(eps)}, "Wb": Wb,
                   "gpu_lock": mp.get_context("fork").Lock()}
    rows = {}
    with mp.get_context("fork").Pool(args.workers) as pool:
        for r in tqdm(pool.imap_unordered(_report_chunk, list(by_chunk.values())), total=len(by_chunk),
                      desc="report chunks", unit="chunk", dynamic_ncols=True):
            rows.update(r)
    df = pd.DataFrame([rows[ci] for ci in sorted(rows)])
    df.to_csv(out / "sweep_report.csv", index=False)
    pd.set_option("display.width", 220)
    print(df[["name", "k", "samples", "benign_AP50", "attack_AP50", "no_att_AP50", "drop50_pct"]].round(2).to_string(index=False))


def main():
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap_.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--frames-work", required=True)
    r.add_argument("--configs", required=True)
    r.add_argument("--box-order", choices=["hwl", "lwh"], default="hwl", help="box size order of the detector")
    r.add_argument("--nms-thresh", type=float, default=0.15, help="rotated-NMS IoU threshold of late fusion")
    r.add_argument("--episodes", nargs="*", default=None)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--workers", type=int, default=20)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--overwrite", action="store_true")
    r.add_argument("--chunk", type=int, default=6, help="configurations per task")
    p = sub.add_parser("report")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--workers", type=int, default=4, help="chunks processed in parallel (memory: a few GB each)")
    a = ap_.parse_args()
    {"run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
