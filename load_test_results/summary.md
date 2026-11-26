## Flood test results: `/predict` (100 concurrent users, 60s per run)

| Containers | Requests | Failures | Throughput (req/s) | Avg (ms) | Median (ms) | p95 (ms) | p99 (ms) | Max (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1108 | 0 | 19.5 | 3323 | 3100 | 5700 | 6200 | 7019 |
| 2 | 2742 | 0 | 46.0 | 853 | 510 | 3400 | 4200 | 5515 |
| 4 | 3781 | 0 | 65.2 | 297 | 170 | 1100 | 1600 | 2671 |
