# Phase 3 Best Points (All Systems)

Generated: 20260204_133915

| System | Baseline Services | Baseline Q | Baseline IFN_Ratio | CAC Services | CAC Q | CAC IFN_Ratio | dpep_cap | target_range |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| acmeair | - | 0.0000 | 0.0000 | 5 | 0.0855 | 0.3697 | 0.14 | [5,5] |
| daytrader | 4 | 0.0417 | 0.6879 | 7 | 0.1647 | 0.5828 | 0.18 | [5,8] |
| plants | 3 | 0.0514 | 0.6682 | 5 | 0.0878 | 0.6119 | 0.22 | [3,6] |
| jpetstore | 4 | 0.0279 | 0.7267 | 4 | 0.0339 | 0.6286 | 0.14 | [4,8] |

## Raw stdout tails (debug)

### acmeair

```
[Phase3] phase3_cac_evaluation.py starting...
[Phase3] args: systems=['acmeair'] mode=sigmoid dpep_cap=0.14 merge_small_clusters=True min_cluster_size=3
System          | Services   | Modularity | Result    
-------------------------------------------------------
[Phase3] Running system=acmeair target_range_override=(5,5)
[UniversePolicy] acmeair: StrategyA enabled | kept=41 / original=41
[GraphPolicy] acmeair: DPEP enabled | p=70 cap=0.140 => tau=0.140000 | final edge_min_weight=0.140000

=== [Phase 3] Evaluating CAC Algorithm for acmeair ===

[GraphDiag] acmeair | nodes=41 | edge_min_weight=0.14 | Baseline edges=552 (density=0.6732) | CAC edges=238 (density=0.2902) | mode=sigmoid k=6.0 n_power=4.0 alpha=15.0 beta=0.14298115432623462
[ClusterSizes] acmeair | CAC-Final | before_merge | clusters=5 nodes=41 min/p25/med/p75/max=1/1/1/13/25 singletons=3 top10=[25, 13, 1, 1, 1]
[ClusterSizes] acmeair | CAC-Final | after_merge | clusters=5 nodes=41 min/p25/med/p75/max=1/1/1/13/25 singletons=3 top10=[25, 13, 1, 1, 1]
[INFO] 参数已保存: data/processed/fusion/acmeair_cac_params.json

============================================================
Method          | Q (↑)    | IFN (↓)  | IFN_Ratio (↓)  | Services  
---------------------------------------------------------------------------
Baseline        | No valid partition found in range (5, 5).
CAC-Final       | 0.0855   | 88       | 0.3697         | 5         
============================================================
acmeair         | 5          | 0.0855     | PASS      
[INFO] 标准化输出: data/processed/fusion/acmeair_baseline_partition.json
[INFO] 标准化输出: data/processed/fusion/acmeair_cac-final_partition.json

% LaTeX Table Code
\begin{table}[h]
\centering
\begin{tabular}{lcc}
\hline
System & Services & Modularity ($Q$) \
\hline
Acmeair & 5 & 0.0855 \
\hline
\end{tabular}
\caption{Performance of Uncertainty-Aware Clustering (CAC)}
\end{table}

```

### daytrader

```
[Phase3] phase3_cac_evaluation.py starting...
[Phase3] args: systems=['daytrader'] mode=sigmoid dpep_cap=0.18 merge_small_clusters=True min_cluster_size=3
System          | Services   | Modularity | Result    
-------------------------------------------------------
[Phase3] Running system=daytrader target_range_override=(5,8)
[UniversePolicy] daytrader: StrategyA enabled | kept=34 / original=34
[GraphPolicy] daytrader: DPEP enabled | p=70 cap=0.180 => tau=0.180000 | final edge_min_weight=0.180000

=== [Phase 3] Evaluating CAC Algorithm for daytrader ===

[GraphDiag] daytrader | nodes=34 | edge_min_weight=0.18 | Baseline edges=503 (density=0.8966) | CAC edges=163 (density=0.2906) | mode=sigmoid k=6.0 n_power=4.0 alpha=15.0 beta=0.08238982408950239
[ClusterSizes] daytrader | Baseline | before_merge | clusters=5 nodes=34 min/p25/med/p75/max=1/3/7/9/14 singletons=1 top10=[14, 9, 7, 3, 1]
[ClusterSizes] daytrader | Baseline | after_merge | clusters=4 nodes=34 min/p25/med/p75/max=3/6/8/10/15 singletons=0 top10=[15, 9, 7, 3]
[ClusterSizes] daytrader | CAC-Final | before_merge | clusters=7 nodes=34 min/p25/med/p75/max=1/1/5/8/10 singletons=3 top10=[10, 10, 6, 5, 1, 1, 1]
[ClusterSizes] daytrader | CAC-Final | after_merge | clusters=7 nodes=34 min/p25/med/p75/max=1/1/5/8/10 singletons=3 top10=[10, 10, 6, 5, 1, 1, 1]
[INFO] 参数已保存: data/processed/fusion/daytrader_cac_params.json

============================================================
Method          | Q (↑)    | IFN (↓)  | IFN_Ratio (↓)  | Services  
---------------------------------------------------------------------------
Baseline        | 0.0417   | 346      | 0.6879         | 4         
CAC-Final       | 0.1647   | 95       | 0.5828         | 7         
============================================================
daytrader       | 7          | 0.1647     | PASS      
[INFO] 标准化输出: data/processed/fusion/daytrader_baseline_partition.json
[INFO] 标准化输出: data/processed/fusion/daytrader_cac-final_partition.json

% LaTeX Table Code
\begin{table}[h]
\centering
\begin{tabular}{lcc}
\hline
System & Services & Modularity ($Q$) \
\hline
Daytrader & 7 & 0.1647 \
\hline
\end{tabular}
\caption{Performance of Uncertainty-Aware Clustering (CAC)}
\end{table}

```

### plants

```
[Phase3] phase3_cac_evaluation.py starting...
[Phase3] args: systems=['plants'] mode=sigmoid dpep_cap=0.22 merge_small_clusters=True min_cluster_size=3
System          | Services   | Modularity | Result    
-------------------------------------------------------
[Phase3] Running system=plants target_range_override=(3,6)
[UniversePolicy] plants: StrategyA enabled | kept=31 / original=31
[GraphPolicy] plants: DPEP enabled | p=70 cap=0.220 => tau=0.220000 | final edge_min_weight=0.220000

=== [Phase 3] Evaluating CAC Algorithm for plants ===

[GraphDiag] plants | nodes=31 | edge_min_weight=0.22 | Baseline edges=422 (density=0.9075) | CAC edges=201 (density=0.4323) | mode=sigmoid k=6.0 n_power=4.0 alpha=15.0 beta=0.06881445526615075
[ClusterSizes] plants | Baseline | before_merge | clusters=3 nodes=31 min/p25/med/p75/max=9/9/10/11/12 singletons=0 top10=[12, 10, 9]
[ClusterSizes] plants | Baseline | after_merge | clusters=3 nodes=31 min/p25/med/p75/max=9/9/10/11/12 singletons=0 top10=[12, 10, 9]
[ClusterSizes] plants | CAC-Final | before_merge | clusters=5 nodes=31 min/p25/med/p75/max=1/4/6/9/11 singletons=1 top10=[11, 9, 6, 4, 1]
[ClusterSizes] plants | CAC-Final | after_merge | clusters=5 nodes=31 min/p25/med/p75/max=1/4/6/9/11 singletons=1 top10=[11, 9, 6, 4, 1]
[INFO] 参数已保存: data/processed/fusion/plants_cac_params.json

============================================================
Method          | Q (↑)    | IFN (↓)  | IFN_Ratio (↓)  | Services  
---------------------------------------------------------------------------
Baseline        | 0.0514   | 282      | 0.6682         | 3         
CAC-Final       | 0.0878   | 123      | 0.6119         | 5         
============================================================
plants          | 5          | 0.0878     | PASS      
[INFO] 标准化输出: data/processed/fusion/plants_baseline_partition.json
[INFO] 标准化输出: data/processed/fusion/plants_cac-final_partition.json

% LaTeX Table Code
\begin{table}[h]
\centering
\begin{tabular}{lcc}
\hline
System & Services & Modularity ($Q$) \
\hline
Plants & 5 & 0.0878 \
\hline
\end{tabular}
\caption{Performance of Uncertainty-Aware Clustering (CAC)}
\end{table}

```

### jpetstore

```
[Phase3] phase3_cac_evaluation.py starting...
[Phase3] args: systems=['jpetstore'] mode=sigmoid dpep_cap=0.14 merge_small_clusters=True min_cluster_size=3
System          | Services   | Modularity | Result    
-------------------------------------------------------
[Phase3] Running system=jpetstore target_range_override=(4,8)
[UniversePolicy] jpetstore: StrategyA enabled | kept=73 / original=73
[GraphPolicy] jpetstore: DPEP enabled | p=70 cap=0.140 => tau=0.140000 | final edge_min_weight=0.140000

=== [Phase 3] Evaluating CAC Algorithm for jpetstore ===

[GraphDiag] jpetstore | nodes=73 | edge_min_weight=0.14 | Baseline edges=2547 (density=0.9692) | CAC edges=1855 (density=0.7059) | mode=sigmoid k=6.0 n_power=4.0 alpha=15.0 beta=0.14116176241012254
[ClusterSizes] jpetstore | Baseline | before_merge | clusters=4 nodes=73 min/p25/med/p75/max=11/15/17/19/28 singletons=0 top10=[28, 17, 17, 11]
[ClusterSizes] jpetstore | Baseline | after_merge | clusters=4 nodes=73 min/p25/med/p75/max=11/15/17/19/28 singletons=0 top10=[28, 17, 17, 11]
[ClusterSizes] jpetstore | CAC-Final | before_merge | clusters=4 nodes=73 min/p25/med/p75/max=10/12/14/20/35 singletons=0 top10=[35, 15, 13, 10]
[ClusterSizes] jpetstore | CAC-Final | after_merge | clusters=4 nodes=73 min/p25/med/p75/max=10/12/14/20/35 singletons=0 top10=[35, 15, 13, 10]
[INFO] 参数已保存: data/processed/fusion/jpetstore_cac_params.json

============================================================
Method          | Q (↑)    | IFN (↓)  | IFN_Ratio (↓)  | Services  
---------------------------------------------------------------------------
Baseline        | 0.0279   | 1851     | 0.7267         | 4         
CAC-Final       | 0.0339   | 1166     | 0.6286         | 4         
============================================================
jpetstore       | 4          | 0.0339     | PASS      
[INFO] 标准化输出: data/processed/fusion/jpetstore_baseline_partition.json
[INFO] 标准化输出: data/processed/fusion/jpetstore_cac-final_partition.json

% LaTeX Table Code
\begin{table}[h]
\centering
\begin{tabular}{lcc}
\hline
System & Services & Modularity ($Q$) \
\hline
Jpetstore & 4 & 0.0339 \
\hline
\end{tabular}
\caption{Performance of Uncertainty-Aware Clustering (CAC)}
\end{table}

```
