# Mono baselines vs ours (acmeair)

Generated: 20260204_223104

| Method | BCubedF1 | MoJoSim | K | GT_K | K-Diff | mu_override | U |
|---|---:|---:|---:|---:|---:|---:|---|
| PureSemantic | 0.4278 | 30.43 | 4 | 4 | +0 | 0.0 | with_u |
| PureStructural | 0.4006 | 47.83 | 4 | 4 | +0 | 1.0 | with_u |
| SimpleFusion_noU | 0.4006 | 47.83 | 4 | 4 | +0 | - | no_u |
| Ours_CAC_withU | 0.4006 | 47.83 | 4 | 4 | +0 | - | with_u |

## Notes
- PureSemantic/PureStructural are approximated via mu_override=0/1 using existing S_sem and S_struct.
- SimpleFusion_noU is CAC with U≡0 (uncertainty disabled) over persisted S_final.
- K-lock uses --target_from_gt to keep service granularity comparable.
