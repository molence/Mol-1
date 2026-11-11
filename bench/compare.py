"""Compare two bench outputs: per-sample Kabsch RMSD and confidence deltas.

Usage: python bench/compare.py bench/results/<ref> bench/results/<new> [more...]
"""

import sys

import torch


def kabsch_rmsd(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    u, _, vt = torch.linalg.svd(a.T @ b)
    d = torch.sign(torch.linalg.det(u @ vt))
    s = torch.diag(torch.tensor([1.0, 1.0, d]))
    r = u @ s @ vt
    return ((a @ r - b) ** 2).sum(-1).mean().sqrt().item()


def summ(p, key):
    s = p["summary"]
    v = s[key] if isinstance(s, dict) else [x[key] for x in s]
    return torch.tensor(v, dtype=torch.float64).flatten()


def main():
    ref = torch.load(sys.argv[1] + "/pred.pt")
    for path in sys.argv[2:]:
        new = torch.load(path + "/pred.pt")
        x0, x1 = ref["coordinate"].double(), new["coordinate"].double()
        rmsd = [kabsch_rmsd(x0[i], x1[i]) for i in range(x0.shape[0])]
        line = [f"{path.rstrip('/').split('/')[-1]:>24s}", "rmsd/sample " + " ".join(f"{r:.3f}" for r in rmsd)]
        for k in ("plddt", "ptm", "iptm", "ranking_score"):
            a, b = summ(ref, k), summ(new, k)
            line.append(f"{k} {b.mean():.4f} (d {(b - a).abs().max():.4f})")
        for k in ("plddt", "pae", "contact_probs"):
            if k in ref and k in new:
                pa, pb = ref[k].double(), new[k].double()
                if k != "contact_probs":
                    pa, pb = pa.softmax(-1), pb.softmax(-1)
                d = (pa - pb).abs()
                line.append(f"{k} max {d.max():.3g} mean {d.mean():.2g}")
        print(" | ".join(line))


if __name__ == "__main__":
    main()
