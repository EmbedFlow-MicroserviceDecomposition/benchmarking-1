"""Adapter: our extractor output (nodes.csv + edges.csv) -> VAE-C input.

VAE-C (tools/vaec, rokinmaharjan/benchmarking-monolith-to-microservices) takes
a plain adjacency-list JSON {className: [dependentClassName, ...], ...}
(machine-learning/ae.ipynb cell reading "<app>_dependency.json") plus a list
of entrypoint controller *simple* class names (matched via Python `in`
substring against full class names in the notebook, so simple names are
sufficient and match our own adapters/configs/{app}/entrypoints.txt FQNs by
taking the last path segment).

Usage:
    python3 adapters/vaec_adapter.py --nodes n.csv --edges e.csv \
        --entrypoints eps.txt --out <app>_dependency.json
"""

import argparse
import csv
import json
from collections import defaultdict


def load_nodes(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()][1:]


def load_edges(path):
    with open(path, newline="") as f:
        return [(r["src"], r["dst"]) for r in csv.DictReader(f)]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nodes", required=True)
    p.add_argument("--edges", required=True)
    p.add_argument("--entrypoints", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--entrypoints-out", required=True)
    args = p.parse_args(argv)

    nodes = load_nodes(args.nodes)
    edges = load_edges(args.edges)
    node_set = set(nodes)

    deps = defaultdict(list)
    for n in nodes:
        deps[n] = []
    seen = defaultdict(set)
    for src, dst in edges:
        if src not in node_set or dst not in node_set or src == dst:
            continue
        if dst not in seen[src]:
            seen[src].add(dst)
            deps[src].append(dst)

    with open(args.out, "w") as f:
        json.dump(deps, f, indent=1)

    with open(args.entrypoints) as f:
        entrypoint_fqns = [line.strip() for line in f if line.strip()]
    with open(args.entrypoints_out, "w") as f:
        json.dump(entrypoint_fqns, f, indent=1)

    print("classes=%d edges=%d entrypoints=%d" % (len(nodes), sum(len(v) for v in deps.values()),
                                                    len(entrypoint_fqns)))


if __name__ == "__main__":
    main()
