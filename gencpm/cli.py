"""Command line of GenCPM: `python -m gencpm <command> [options]`. Run `python -m gencpm <command> --help` for the
options of one command. The reproduce/ scripts show the commands in the order that rebuilds the paper's results."""

import argparse
import importlib
import os
import sys

# commands implemented by a module with its own command line (module, function)
PASSTHROUGH = {
    "generate": ("gencpm.cpm.generate", "main", "build benign and attacked CPMs from the released frames"),
    "coordinated": ("gencpm.attacks.coordinated_set", "main", "build the coordinated-attack collection"),
    "summarize": ("gencpm.evaluation.summary", "main", "detection metrics of every detector in a work dir"),
    "fusion": ("gencpm.evaluation.fusion", "main", "end-to-end late fusion: run | report | robosac-calib"),
    "sweep": ("gencpm.evaluation.sweeps", "main", "attack-parameter and number-of-attackers sweeps: run | report"),
    "table-main": ("gencpm.report.main_tables", "main", "Table 8"),
    "table-distribution": ("gencpm.report.distribution", "main", "Table 6"),
    "table-second": ("gencpm.report.second_table", "main", "Table 9"),
    "table-cooperscene": ("gencpm.report.cooperscene_table", "main", "Table 10"),
    "figures": ("gencpm.report.figures", "main", "ROC curves and window-length figure"),
    "figures-appendix": ("gencpm.report.appendix_figures", "main", "number-of-attackers and sensitivity figures"),
    "download": ("gencpm.download", "main", "download and unpack the AdverCPM release from Hugging Face"),
    "paths": ("gencpm.paths", "main", "show the configured dataset and work paths"),
    "infer": ("gencpm.perception.infer", "main", "optional: LiDAR detection for a new dataset (needs OpenCOOD)"),
    "cooperscene-layout": ("gencpm.perception.cooperscene", "main", "optional: CooperScene detections -> frames layout"),
}


def _common(p):
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--overwrite", action="store_true")


def _parser():
    ap = argparse.ArgumentParser(prog="python -m gencpm", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="command")
    workers = max(1, (os.cpu_count() or 2) - 2)

    def cmd(name, help_):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--work-dir", required=True, help="work directory of one collection")
        return p

    f = cmd("frames", "cache the released frames (detections, scores, ground truth, poses)")
    f.add_argument("--data-root", required=True, help="released frames: <root>/Town*/<episode>/<vehicle>/*_preds.yaml")
    f.add_argument("--workers", type=int, default=workers)
    c = cmd("cache", "cache the generated CPMs with content labels")
    c.add_argument("--cpm-root", required=True)
    c.add_argument("--label-rule", choices=["content", "schedule"], default="content",
                   help="content: malicious only if the CPM differs from what the sender perceived (default)")
    c.add_argument("--workers", type=int, default=workers)
    cmd("features", "receiver-side object and CPM features")
    fi = cmd("fit", "fit rules, PILD and gradient-boosted trees per fold")
    for p in (fi,):
        p.add_argument("--folds", type=int, default=5)
        p.add_argument("--split-seed", type=int, default=0)
        p.add_argument("--seq-len", type=int, default=20)
        p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    fi.add_argument("--pretrained", default="", help="load PILD from this defenses_W*/ dir instead of training it")
    tr = cmd("transfer", "apply PILD and gradient-boosted trees trained on other work dirs")
    tr.add_argument("--sources", nargs="+", default=[], help="work dirs to train on (same episodes)")
    tr.add_argument("--pretrained", default="",
                    help="instead of --sources: work dir (or released model dir) whose saved fit/baselines models to load")
    tr.add_argument("--folds", type=int, default=5)
    tr.add_argument("--split-seed", type=int, default=0)
    tr.add_argument("--seq-len", type=int, default=20)
    tr.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    t = cmd("train", "train the temporal models (LSTM, GRU, BiLSTM, CNN-LSTM) per fold"); _common(t)
    t.add_argument("--max-objects", type=int, default=35)
    t.add_argument("--coords", choices=["global", "ego"], default="global",
                   help="global = paper representation; ego = ego-relative positions")
    t.add_argument("--models", default="", help="comma list, default all four")
    t.add_argument("--epochs", type=int, default=70)
    t.add_argument("--patience", type=int, default=8)
    t.add_argument("--batch-size", type=int, default=128)
    t.add_argument("--verbose", action="store_true")
    t.add_argument("--cpu", action="store_true", help="force CPU")
    t.add_argument("--only-folds", type=int, nargs="+", default=None)
    a = cmd("apply", "apply trained temporal models to another work dir"); _common(a)
    a.add_argument("--source-dir", required=True, help="temporal_W*/ dir of the work dir the models were trained in")
    a.add_argument("--max-objects", type=int, default=35)
    a.add_argument("--coords", choices=["global", "ego"], default="global")
    a.add_argument("--models", default="")
    a.add_argument("--out-name", default="", help="output dir name (default temporal_W<L>_<coords>)")
    a.add_argument("--tag-suffix", default="", help="appended to the model name, e.g. -transfer")
    b = cmd("baselines", "rule baselines and gradient-boosted trees on rule features"); _common(b)
    b.add_argument("--corroboration-radius", type=float, default=2.0)
    b.add_argument("--sensing-range", type=float, default=50.0)
    b.add_argument("--pretrained", default="", help="load the trees from this baselines_W*/ dir instead of training them")
    for name, (_, _, help_) in PASSTHROUGH.items():
        sub.add_parser(name, help=help_, add_help=False)
    return ap


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in PASSTHROUGH:
        mod, fn, _ = PASSTHROUGH[argv[0]]
        sys.argv = [f"gencpm {argv[0]}"] + argv[1:]
        return getattr(importlib.import_module(mod), fn)()
    args = _parser().parse_args(argv)
    from gencpm.data import cache
    from gencpm.defenses import baselines, features, pild, temporal, transfer
    {"frames": cache.cmd_frames, "cache": cache.cmd_cache, "features": features.cmd_features, "fit": pild.cmd_fit,
     "transfer": transfer.cmd_transfer, "train": temporal.cmd_train, "apply": temporal.cmd_apply,
     "baselines": baselines.cmd_baselines}[args.cmd](args)
