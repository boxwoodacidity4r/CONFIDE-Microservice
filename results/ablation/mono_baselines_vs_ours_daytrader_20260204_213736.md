# Mono baselines vs ours (daytrader)

Generated: 20260204_213736

| Method | BCubedF1 | MoJoSim | K | GT_K | K-Diff | mu_override | U |
|---|---:|---:|---:|---:|---:|---:|---|
| PureSemantic | 0.3780 | 28.12 | 0 | 0 | +0 | 0.0 | with_u |
| PureStructural | 0.3780 | 28.12 | 0 | 0 | +0 | 1.0 | with_u |
| SimpleFusion_noU | 0.4972 | 46.88 | 0 | 0 | +0 | - | no_u |
| Ours_CAC_withU | 0.5491 | 50.00 | 0 | 0 | +0 | 0.3 | with_u |

## Notes
- PureSemantic/PureStructural are approximated via mu_override=0/1 using existing S_sem and S_struct.
- SimpleFusion_noU is CAC with U≡0 (uncertainty disabled) over persisted S_final.
- K-lock uses --target_from_gt to keep service granularity comparable.
