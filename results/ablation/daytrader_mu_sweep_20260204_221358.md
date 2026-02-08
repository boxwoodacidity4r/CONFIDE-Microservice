# DayTrader mu_override sweep (surgical weight tuning)

Generated: 20260204_221358

Fixed cap=0.16, mode=sigmoid, alpha=15.0, merge_small_clusters=True, min_cluster_size=3

| mu_override | BCubedF1 | MoJoSim | K | GT_K | K-Diff |
|---:|---:|---:|---:|---:|---:|
| 0.30 | 0.5491 | 50.00 | 5 | 5 | +0 |
| 0.20 | 0.4873 | 53.12 | 6 | 5 | +1 |
| 0.40 | 0.4050 | 43.75 | 6 | 5 | +1 |
| 0.80 | 0.4050 | 43.75 | 6 | 5 | +1 |
| 0.90 | 0.4050 | 43.75 | 6 | 5 | +1 |
| 0.70 | 0.4050 | 40.62 | 6 | 5 | +1 |
| 0.50 | 0.4050 | 40.62 | 6 | 5 | +1 |
| 0.60 | 0.4050 | 40.62 | 6 | 5 | +1 |
| 0.10 | 0.3832 | 40.62 | 5 | 5 | +0 |

## K-collapse diagnostic
If Baseline collapses to very small K (e.g., 2) while GT_K is larger, its score can be misleading. This sweep reports K for each mu override so CAC can be compared under similar granularity.

**Best by BCubedF1:**

{
  "mu_override": 0.3,
  "cap": 0.16,
  "bcubed_f1": 0.5490990990990992,
  "mojosim": 50.0,
  "pred_k": 5,
  "gt_k": 5,
  "k_diff": 0
}
