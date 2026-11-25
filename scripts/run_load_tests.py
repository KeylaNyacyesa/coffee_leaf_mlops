"""Flood the model with requests using 1, 2 and 4 Docker containers and record the latency.

Run from the project root (Windows PowerShell, macOS or Linux), with Docker Desktop running:

    python scripts/run_load_tests.py
    python scripts/run_load_tests.py --users 200 --duration 90s --containers 1 2 3 4

For each number of containers it:
    1. starts that many API containers behind nginx     (docker compose up --scale api=N)
    2. waits until nginx is spreading requests over all N containers
    3. runs Locust headless against http://localhost:8080
    4. saves load_test_results/<N>_containers_stats.csv
Finally it writes load_test_results/summary.md and load_test_results/latency_vs_containers.png.
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "load_test_results"
HOST = "http://localhost:8080"


def run(command: list[str]) -> None:
    print(">", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def wait_for_containers(n: int, timeout: int = 300) -> None:
    """Call /health repeatedly until N different containers have answered through nginx."""
    seen, deadline = set(), time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{HOST}/health", timeout=5) as response:
                body = json.load(response)
            if body["status"] == "ok":
                seen.add(body["container"])
        except Exception:          # noqa: BLE001  (containers still starting)
            pass
        if len(seen) >= n:
            print(f"  all {n} container(s) are answering: {sorted(seen)}", flush=True)
            return
        time.sleep(0.5)
    raise SystemExit(f"Only {len(seen)} of {n} containers answered within {timeout} s. Check `docker compose logs api`.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--containers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--users", type=int, default=100, help="concurrent simulated users")
    parser.add_argument("--spawn-rate", type=int, default=10, help="users started per second")
    parser.add_argument("--duration", default="60s", help="length of each run, e.g. 60s or 2m")
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    run(["docker", "compose", "build"])

    for n in args.containers:
        print(f"\n===== {n} container(s) =====", flush=True)
        run(["docker", "compose", "up", "-d", "--scale", f"api={n}"])
        wait_for_containers(n)
        time.sleep(5)              # let the last container finish its warm-up
        run([sys.executable, "-m", "locust", "-f", "locustfile.py", "--host", HOST, "--headless",
             "-u", str(args.users), "-r", str(args.spawn_rate), "-t", args.duration,
             "--csv", str(RESULTS / f"{n}_containers"), "--only-summary"])

    sys.path.insert(0, str(ROOT / "scripts"))
    from summarize_load_tests import summarize
    summarize(users=args.users, duration=args.duration)
    print(f"\nDone. Results are in {RESULTS / 'summary.md'} and latency_vs_containers.png")


if __name__ == "__main__":
    main()
