## Flood test results: `/predict` (100 concurrent users, 60s per run)

| Containers | Requests | Failures | Throughput (req/s) | Avg (ms) | Median (ms) | p95 (ms) | p99 (ms) | Max (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 502 | 0 | 8.6 | 8862 | 10000 | 12000 | 15000 | 16641 |
| 2 | 1134 | 0 | 19.3 | 3352 | 2300 | 8200 | 8600 | 12012 |
| 4 | 2235 | 0 | 37.4 | 1306 | 1200 | 3000 | 3900 | 6919 |
