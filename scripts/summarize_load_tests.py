"""Turn Locust CSVs (load_test_results/<n>_containers_stats.csv) into a markdown table + chart."""
import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "load_test_results"


def summarize(users="?", duration="?"):
    rows = []
    for f in sorted(OUT.glob("*_containers_stats.csv"), key=lambda p: int(re.match(r"(\d+)", p.name).group(1))):
        n = int(re.match(r"(\d+)", f.name).group(1))
        for r in csv.DictReader(open(f, newline="")):
            if r["Name"] == "/predict":
                rows.append({
                    "containers": n, "requests": int(r["Request Count"]), "failures": int(r["Failure Count"]),
                    "rps": float(r["Requests/s"]), "avg": float(r["Average Response Time"]),
                    "median": float(r["Median Response Time"]), "p95": float(r["95%"]), "p99": float(r["99%"]),
                    "max": float(r["Max Response Time"]),
                })
    if not rows:
        raise SystemExit("No Locust CSVs found in load_test_results/")

    md = [f"## Flood test results: `/predict` ({users} concurrent users, {duration} per run)\n",
          "| Containers | Requests | Failures | Throughput (req/s) | Avg (ms) | Median (ms) | p95 (ms) | p99 (ms) | Max (ms) |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['containers']} | {r['requests']} | {r['failures']} | {r['rps']:.1f} | {r['avg']:.0f} | "
                  f"{r['median']:.0f} | {r['p95']:.0f} | {r['p99']:.0f} | {r['max']:.0f} |")
    (OUT / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    n = [r["containers"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, lab in [("median", "median"), ("avg", "average"), ("p95", "95th percentile")]:
        ax[0].plot(n, [r[k] for r in rows], marker="o", label=lab)
    ax[0].set_xlabel("Docker containers"); ax[0].set_ylabel("response time (ms)"); ax[0].set_xticks(n)
    ax[0].set_title("Latency vs number of containers"); ax[0].legend()
    ax[1].bar([str(x) for x in n], [r["rps"] for r in rows], color="#2f7d32")
    ax[1].set_xlabel("Docker containers"); ax[1].set_ylabel("requests / second"); ax[1].set_title("Throughput")
    plt.tight_layout(); plt.savefig(OUT / "latency_vs_containers.png", dpi=130)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", default="?")
    ap.add_argument("--duration", default="?")
    a = ap.parse_args()
    summarize(a.users, a.duration)
