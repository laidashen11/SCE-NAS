
import argparse
import json
import os
import shutil

import numpy as np

from genotypesNAS101 import Structure

def iter_modules(api, epochs, stats=None):
    
    stats = stats if stats is not None else {}
    for h in api.hash_iterator():
        try:
            fixed = api.fixed_statistics[h]
            comp = api.computed_statistics[h]
        except AttributeError:
            fixed, comp = api.get_metrics_from_hash(h)
        adj = np.asarray(fixed['module_adjacency'])
        dim = adj.shape[0] if adj.ndim == 2 else -1
        stats[dim] = stats.get(dim, 0) + 1
        if dim != 7:
            continue
        dpoints = comp.get(epochs)
        if not dpoints:
            continue
        valid = float(np.mean([dp['final_validation_accuracy'] for dp in dpoints]))
        test = float(np.mean([dp['final_test_accuracy'] for dp in dpoints]))
        yield valid, test, adj, list(fixed['module_operations'])

def to_str(adj, ops7):
    return Structure(adj, ops7[1:6]).tostr()

def parse_str(s):
    ops_part, edges_part = s.split('+', 1)
    ops = [o for o in ops_part.strip('|').split('|') if o]
    adj = np.zeros((7, 7), dtype=np.int32)
    for e in edges_part.strip('|').split(';'):
        e = e.strip()
        if not e:
            continue
        x, y = e.split('->')
        adj[int(x), int(y)] = 1
    return adj, ops

def accs_from_str(api, s, epochs):
    
    from nasbench.lib.model_spec import ModelSpec
    adj, ops = parse_str(s)
    spec = ModelSpec(matrix=adj, ops=['input'] + list(ops) + ['output'])
    _, comp = api.get_metrics_from_spec(spec)
    dp = comp[epochs]
    return (float(np.mean([d['final_validation_accuracy'] for d in dp])),
            float(np.mean([d['final_test_accuracy'] for d in dp])))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--api_path', default='./nasbench_only108.tfrecord')
    ap.add_argument('--old_fewshot', default='./fewshot_archs101.json')
    ap.add_argument('--out', default='./fewshot_archs101.json')
    ap.add_argument('--epochs', type=int, default=108)
    ap.add_argument('--p_lo', type=float, default=50.0, help='lower valid quantile for the 15 tail examples')
    a = ap.parse_args()

    from nasbench import api as nasbench_api
    print("[make_fewshot101_v2] valid-only; max = global valid optimum (used by best_static)")
    api = nasbench_api.NASBench(a.api_path)

    valid, test, str_of = [], [], []
    str2acc = {}
    n = 0
    stats = {}
    for v, t, adj, ops7 in iter_modules(api, a.epochs, stats=stats):
        s = to_str(adj, ops7)
        valid.append(v); test.append(t); str_of.append(s)
        str2acc[s] = (v, t)
        n += 1
        if n % 50000 == 0:
            print("  enumerated {} ...".format(n))
    valid = np.asarray(valid); test = np.asarray(test)
    print("[enumerate] 7x7 (in-space) modules: {} ; dim histogram: {}".format(n, dict(sorted(stats.items()))))
    if n == 0:
        raise SystemExit("no 7x7 modules enumerated; check the tfrecord / nasbench version")
    print("[dist] valid: min={:.4f} max={:.4f} p50={:.4f} p90={:.4f}".format(
        valid.min(), valid.max(), np.percentile(valid, 50), np.percentile(valid, 90)))
    print("[dist] test : min={:.4f} max={:.4f} (for reference only, not written)".format(test.min(), test.max()))

    with open(a.old_fewshot, encoding='utf-8') as f:
        old = json.load(f)
    low5_str = [e['genotype_str'] for e in old[:5]]
    low5 = []
    for s in low5_str:
        if s in str2acc:
            v, _ = str2acc[s]
        else:
            v, _ = accs_from_str(api, s, a.epochs)
            print("[warn] old low anchor not in the 7x7 enumeration, fallback to direct query: {}".format(s))
        low5.append({"genotype_str": s, "true_acc_valid": v})

    used = set(low5_str)
    order = sorted(range(n), key=lambda i: valid[i])              
    v_lo = np.percentile(valid, a.p_lo)
    cand = [i for i in order if valid[i] >= v_lo and str_of[i] not in used]
    need = 20 - len(low5)
    idx = np.linspace(0, len(cand) - 1, need).astype(int)         
    pick = []
    for j in idx:
        i = cand[int(j)]
        pick.append(i)
    dedup, seen = [], set()
    for i in pick:
        if i not in seen:
            seen.add(i); dedup.append(i)
    if len(dedup) < need:
        for i in reversed(cand):
            if i not in seen:
                seen.add(i); dedup.append(i)
                if len(dedup) >= need:
                    break
    dedup = sorted(dedup, key=lambda i: valid[i])
    tail = [{"genotype_str": str_of[i], "true_acc_valid": float(valid[i])} for i in dedup]

    out = low5 + tail
    global_best = max(out, key=lambda e: e['true_acc_valid'])
    print("[check] few-shot max valid = {:.6f} (== global valid optimum {:.6f}? {})".format(
        global_best['true_acc_valid'], valid.max(), abs(global_best['true_acc_valid'] - float(valid.max())) < 1e-9))

    if os.path.abspath(a.out) == os.path.abspath(a.old_fewshot) and os.path.exists(a.old_fewshot):
        shutil.copyfile(a.old_fewshot, a.old_fewshot + '.old.json')
        print("[backup] {} -> {}.old.json".format(a.old_fewshot, a.old_fewshot))
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\n[saved] {} entries -> {}".format(len(out), a.out))
    print("idx |  role | valid   | genotype")
    for i, e in enumerate(out):
        role = 'low' if i < len(low5) else 'tail'
        print("{:3d} | {:4s} | {:.6f} | {}".format(i, role, e['true_acc_valid'], e['genotype_str']))

if __name__ == '__main__':
    main()
