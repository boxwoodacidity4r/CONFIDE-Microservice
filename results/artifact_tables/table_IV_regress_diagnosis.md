# Table IV: Multimodal Conflict vs Dynamic Evidence Coverage (Diagnosis)

Suppression count definition: count edges with basic association (S>0.10) whose weight drops by >30% under uncertainty gating.\n\n| System | Dynamic non-zero ratio (NZ%) | Strong-edge suppressed count (S>0.10, drop>30%) | IFN change (withU vs noU) | BCubed F1 change (withU vs noU) | Diagnosis |
|:---|:---:|:---:|:---:|:---:|:---|
| acmeair | 4.17% | 1531 | +3.8% | -0.4% | Stable gain: moderate coverage and conflict structure; uncertainty gating suppresses noisy edges while keeping partitions consistent. |
| daytrader | 0.12% | 3699 | +42.6% | +20.4% | Targeted suppression: although evidence is sparse, U identifies and cuts key cross-domain conflict edges. |
| plants | 32.05% | 457 | +0.0% | +0.0% | Architectural trade-off: suppressing conflict edges reduces coupling but may sacrifice partition purity. |
| jpetstore | 2.06% | 2510 | -30.0% | +5.9% | Sparse-evidence failure mode: dynamic evidence coverage is too low; uncertainty estimates become coverage-limited. |
