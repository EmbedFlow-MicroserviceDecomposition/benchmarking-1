import sys, os, csv, json
from collections import defaultdict
from statistics import mean, stdev

sys.path.insert(0, "/Users/ravishan/Desktop/FYP/fyp-benchmarking")
from metrics.metrics import compute_all, load_edges_csv, Operation, bcp, chm, chd

sys.path.insert(0, "/Users/ravishan/Desktop/FYP/fyp-benchmarking/adapters")
from chgnn_adapter import reachable, _ep_annotation, load_edges as chgnn_load_edges

BASE = "/Users/ravishan/Desktop/FYP/fyp-benchmarking"

APPS = {
    "daytrader": 3,
    "plantsbywebsphere": 3,
}


def load_calls(path):
    calls = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            op = Operation(cls=r["callee"], name=r["method"],
                            params=tuple(r["params"].split(";")) if r["params"] else (),
                            returns=r["returns"])
            calls.append((r["caller"], op))
    return calls


def build_class_usecases(edges, entrypoints):
    class_usecases = defaultdict(list)
    for ep in entrypoints:
        ann = _ep_annotation(ep)
        for cls in reachable(edges, ep):
            class_usecases[cls].append(ann)
    return class_usecases


def invert(clusters_dict):
    partition = {}
    for cid, members in clusters_dict.items():
        for m in members:
            partition[m] = int(cid)
    return partition


all_summaries = {}

for app, n_runs in APPS.items():
    app_dir = os.path.join(BASE, "outputs/graphs", app)
    ep_file = os.path.join(BASE, "adapters/configs", app, "entrypoints.txt")

    edges = load_edges_csv(os.path.join(app_dir, "edges.csv"))
    chgnn_edges = chgnn_load_edges(os.path.join(app_dir, "edges.csv"))
    calls = load_calls(os.path.join(app_dir, "calls.csv"))

    with open(ep_file) as f:
        entrypoints = [l.strip() for l in f if l.strip()]
    class_usecases = build_class_usecases(chgnn_edges, entrypoints)

    results = defaultdict(list)
    chosen_ks = []

    for i in range(1, n_runs + 1):
        fname = os.path.join(BASE, "runs/vaec_repro", app, "result_%d.json" % i)
        with open(fname) as f:
            clusters = json.load(f)
        partition = invert(clusters)
        chosen_ks.append(len(set(partition.values())))

        metrics = compute_all(partition, edges)
        metrics["BCP"] = bcp(partition, class_usecases)
        metrics["CHM"] = chm(partition, calls)
        metrics["CHD"] = chd(partition, calls)

        for k, v in metrics.items():
            results[k].append(v)

    summary = {m: round(mean(vals), 4) for m, vals in results.items()}
    summary["_chosen_ks"] = chosen_ks
    summary["_n_runs"] = n_runs
    all_summaries[app] = summary
    print(app, json.dumps(summary, indent=2))

with open("/tmp/vaec_dt_pbw_summary.json", "w") as f:
    json.dump(all_summaries, f, indent=2)
