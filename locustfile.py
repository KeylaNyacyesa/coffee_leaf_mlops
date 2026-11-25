"""Locust flood test for the prediction API.

Interactive (web UI on http://localhost:8089):
    locust -f locustfile.py --host http://localhost:8080

Headless (what scripts/run_load_tests.sh does):
    locust -f locustfile.py --host http://localhost:8080 --headless -u 100 -r 10 -t 60s --csv load_test_results/run
"""
import random
from pathlib import Path

from locust import HttpUser, between, events, task

TEST_DIR = Path(__file__).parent / "data" / "test"
IMAGES = []


@events.init.add_listener
def load_images(environment, **_):
    files = sorted(p for p in TEST_DIR.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    random.Random(0).shuffle(files)
    for p in files[:200]:
        IMAGES.append((p.name, p.read_bytes(), p.parent.name))
    print(f"Locust: loaded {len(IMAGES)} test images")


class LeafUser(HttpUser):
    wait_time = between(0.5, 1.5)          # a farmer's app sends an image, waits, sends another

    @task(10)
    def predict(self):
        name, data, label = random.choice(IMAGES)
        with self.client.post("/predict", files={"file": (name, data, "image/jpeg")},
                              name="/predict", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
            else:
                r.success()

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
