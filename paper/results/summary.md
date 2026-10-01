
## medquad  (n_queries=15395)

| model | nDCG@10 [95% CI] | Recall@10 | Recall@100 | MRR@10 |
|---|---|---|---|---|
| gte-small | 0.725 [0.719, 0.730] | 0.883 | 0.935 | 0.673 |
| bge-small | 0.723 [0.718, 0.729] | 0.870 | 0.925 | 0.675 |
| e5-small | 0.713 [0.708, 0.719] | 0.865 | 0.927 | 0.664 |
| minilm-base | 0.642 [0.636, 0.648] | 0.832 | 0.921 | 0.581 |
| hybrid-bge-small | 0.626 [0.621, 0.632] | 0.844 | 0.922 | 0.555 |
| hybrid-path-models_ft_avg | 0.561 [0.556, 0.567] | 0.822 | 0.927 | 0.478 |
| path-models_ft_avg | 0.547 [0.541, 0.552] | 0.814 | 0.931 | 0.462 |
| path-models_ft_s2024 | 0.544 [0.538, 0.549] | 0.811 | 0.930 | 0.459 |
| path-models_ft_s13 | 0.543 [0.538, 0.549] | 0.811 | 0.930 | 0.458 |
| path-models_ft_s42 | 0.540 [0.535, 0.546] | 0.808 | 0.928 | 0.456 |
| medcpt | 0.535 [0.529, 0.540] | 0.792 | 0.932 | 0.454 |
| bm25 | 0.502 [0.496, 0.508] | 0.766 | 0.864 | 0.418 |
| hm-3fold | 0.180 [0.175, 0.185] | 0.281 | 0.499 | 0.148 |
| hm-best2 | 0.177 [0.171, 0.182] | 0.276 | 0.495 | 0.146 |

## nfcorpus  (n_queries=323)

| model | nDCG@10 [95% CI] | Recall@10 | Recall@100 | MRR@10 |
|---|---|---|---|---|
| medcpt | 0.353 [0.318, 0.389] | 0.174 | 0.339 | 0.531 |
| gte-small | 0.349 [0.314, 0.384] | 0.169 | 0.337 | 0.541 |
| hybrid-bge-small | 0.344 [0.310, 0.380] | 0.164 | 0.304 | 0.551 |
| bge-small | 0.343 [0.308, 0.379] | 0.162 | 0.311 | 0.529 |
| hybrid-path-models_ft_avg | 0.326 [0.292, 0.362] | 0.154 | 0.304 | 0.527 |
| e5-small | 0.325 [0.292, 0.360] | 0.159 | 0.297 | 0.524 |
| minilm-base | 0.316 [0.282, 0.350] | 0.155 | 0.312 | 0.504 |
| bm25 | 0.306 [0.273, 0.340] | 0.152 | 0.242 | 0.509 |
| path-models_ft_s13 | 0.295 [0.262, 0.329] | 0.141 | 0.292 | 0.484 |
| path-models_ft_s42 | 0.295 [0.262, 0.329] | 0.136 | 0.285 | 0.487 |
| path-models_ft_avg | 0.293 [0.260, 0.327] | 0.137 | 0.285 | 0.482 |
| path-models_ft_s2024 | 0.284 [0.252, 0.318] | 0.133 | 0.281 | 0.467 |
| hm-3fold | 0.069 [0.054, 0.085] | 0.024 | 0.123 | 0.149 |
| hm-best2 | 0.068 [0.053, 0.085] | 0.022 | 0.120 | 0.157 |

## scifact  (n_queries=300)

| model | nDCG@10 [95% CI] | Recall@10 | Recall@100 | MRR@10 |
|---|---|---|---|---|
| medcpt | 0.735 [0.696, 0.774] | 0.881 | 0.975 | 0.695 |
| gte-small | 0.727 [0.684, 0.769] | 0.848 | 0.950 | 0.694 |
| bge-small | 0.713 [0.669, 0.755] | 0.836 | 0.942 | 0.682 |
| hybrid-bge-small | 0.709 [0.664, 0.752] | 0.815 | 0.965 | 0.680 |
| e5-small | 0.687 [0.644, 0.731] | 0.806 | 0.928 | 0.658 |
| hybrid-path-models_ft_avg | 0.656 [0.611, 0.701] | 0.782 | 0.961 | 0.618 |
| bm25 | 0.652 [0.606, 0.696] | 0.774 | 0.873 | 0.619 |
| minilm-base | 0.645 [0.600, 0.689] | 0.783 | 0.925 | 0.605 |
| path-models_ft_s13 | 0.583 [0.537, 0.628] | 0.751 | 0.884 | 0.532 |
| path-models_ft_avg | 0.580 [0.536, 0.626] | 0.740 | 0.888 | 0.531 |
| path-models_ft_s2024 | 0.576 [0.531, 0.622] | 0.736 | 0.874 | 0.529 |
| path-models_ft_s42 | 0.576 [0.531, 0.622] | 0.744 | 0.889 | 0.526 |
| hm-best2 | 0.250 [0.210, 0.290] | 0.365 | 0.604 | 0.220 |
| hm-3fold | 0.249 [0.209, 0.290] | 0.362 | 0.603 | 0.223 |

## trec-covid  (n_queries=50)

| model | nDCG@10 [95% CI] | Recall@10 | Recall@100 | MRR@10 |
|---|---|---|---|---|
| bge-small | 0.756 [0.698, 0.813] | 0.021 | 0.133 | 0.931 |
| e5-small | 0.744 [0.676, 0.809] | 0.020 | 0.136 | 0.896 |
| bm25 | 0.578 [0.498, 0.658] | 0.015 | 0.094 | 0.758 |
| minilm-base | 0.472 [0.389, 0.558] | 0.013 | 0.086 | 0.724 |
| hm-3fold | 0.219 [0.158, 0.286] | 0.005 | 0.038 | 0.374 |

## Paired comparisons (nDCG@10 difference, Holm-adjusted over all rows)

| dataset | A | B | mean(A-B) | 95% CI | p (raw) | p (Holm) |
|---|---|---|---|---|---|---|
| medquad | path-models_ft_s13 | minilm-base | -0.099 | [-0.103, -0.095] | 0.0001 | 0.0046 |
| medquad | e5-small | minilm-base | +0.071 | [+0.067, +0.075] | 0.0001 | 0.0046 |
| medquad | path-models_ft_s42 | minilm-base | -0.102 | [-0.106, -0.098] | 0.0001 | 0.0046 |
| medquad | gte-small | minilm-base | +0.083 | [+0.079, +0.086] | 0.0001 | 0.0046 |
| medquad | path-models_ft_s2024 | minilm-base | -0.098 | [-0.102, -0.095] | 0.0001 | 0.0046 |
| medquad | hybrid-path-models_ft_avg | minilm-base | -0.081 | [-0.086, -0.076] | 0.0001 | 0.0046 |
| medquad | hm-best2 | minilm-base | -0.466 | [-0.472, -0.459] | 0.0001 | 0.0046 |
| medquad | medcpt | minilm-base | -0.108 | [-0.114, -0.101] | 0.0001 | 0.0046 |
| medquad | hybrid-bge-small | minilm-base | -0.016 | [-0.021, -0.011] | 0.0001 | 0.0046 |
| medquad | bge-small | minilm-base | +0.081 | [+0.077, +0.085] | 0.0001 | 0.0046 |
| medquad | bm25 | minilm-base | -0.141 | [-0.147, -0.134] | 0.0001 | 0.0046 |
| medquad | path-models_ft_avg | minilm-base | -0.095 | [-0.099, -0.092] | 0.0001 | 0.0046 |
| medquad | hm-3fold | minilm-base | -0.463 | [-0.469, -0.456] | 0.0001 | 0.0046 |
| medquad | path-models_ft_avg | gte-small | -0.178 | [-0.182, -0.174] | 0.0001 | 0.0046 |
| nfcorpus | hm-3fold | minilm-base | -0.247 | [-0.278, -0.217] | 0.0001 | 0.0046 |
| nfcorpus | gte-small | minilm-base | +0.033 | [+0.018, +0.048] | 0.0001 | 0.0046 |
| nfcorpus | bm25 | minilm-base | -0.010 | [-0.032, +0.013] | 0.4051 | 1.0000 |
| nfcorpus | path-models_ft_avg | minilm-base | -0.023 | [-0.035, -0.011] | 0.0001 | 0.0046 |
| nfcorpus | path-models_ft_s2024 | minilm-base | -0.031 | [-0.044, -0.018] | 0.0001 | 0.0046 |
| nfcorpus | hybrid-path-models_ft_avg | minilm-base | +0.010 | [-0.005, +0.027] | 0.1933 | 0.9665 |
| nfcorpus | hm-best2 | minilm-base | -0.248 | [-0.278, -0.217] | 0.0001 | 0.0046 |
| nfcorpus | e5-small | minilm-base | +0.009 | [-0.006, +0.026] | 0.2417 | 0.9668 |
| nfcorpus | medcpt | minilm-base | +0.037 | [+0.016, +0.060] | 0.0005 | 0.0060 |
| nfcorpus | path-models_ft_s13 | minilm-base | -0.021 | [-0.033, -0.009] | 0.0007 | 0.0077 |
| nfcorpus | bge-small | minilm-base | +0.027 | [+0.013, +0.042] | 0.0007 | 0.0077 |
| nfcorpus | hybrid-bge-small | minilm-base | +0.028 | [+0.009, +0.048] | 0.0035 | 0.0280 |
| nfcorpus | path-models_ft_s42 | minilm-base | -0.021 | [-0.033, -0.009] | 0.0003 | 0.0046 |
| nfcorpus | path-models_ft_avg | medcpt | -0.060 | [-0.085, -0.037] | 0.0001 | 0.0046 |
| scifact | hybrid-bge-small | minilm-base | +0.063 | [+0.029, +0.098] | 0.0011 | 0.0099 |
| scifact | bm25 | minilm-base | +0.007 | [-0.031, +0.044] | 0.7221 | 1.0000 |
| scifact | path-models_ft_s13 | minilm-base | -0.062 | [-0.089, -0.036] | 0.0001 | 0.0046 |
| scifact | bge-small | minilm-base | +0.068 | [+0.041, +0.094] | 0.0001 | 0.0046 |
| scifact | hm-3fold | minilm-base | -0.396 | [-0.442, -0.348] | 0.0001 | 0.0046 |
| scifact | path-models_ft_s42 | minilm-base | -0.069 | [-0.096, -0.041] | 0.0001 | 0.0046 |
| scifact | medcpt | minilm-base | +0.090 | [+0.057, +0.124] | 0.0001 | 0.0046 |
| scifact | e5-small | minilm-base | +0.042 | [+0.014, +0.071] | 0.0049 | 0.0343 |
| scifact | hybrid-path-models_ft_avg | minilm-base | +0.011 | [-0.020, +0.042] | 0.4861 | 1.0000 |
| scifact | path-models_ft_s2024 | minilm-base | -0.069 | [-0.096, -0.040] | 0.0001 | 0.0046 |
| scifact | path-models_ft_avg | minilm-base | -0.065 | [-0.092, -0.036] | 0.0001 | 0.0046 |
| scifact | gte-small | minilm-base | +0.082 | [+0.054, +0.111] | 0.0001 | 0.0046 |
| scifact | hm-best2 | minilm-base | -0.395 | [-0.441, -0.348] | 0.0001 | 0.0046 |
| scifact | path-models_ft_avg | medcpt | -0.154 | [-0.194, -0.115] | 0.0001 | 0.0046 |
| trec-covid | e5-small | minilm-base | +0.271 | [+0.199, +0.345] | 0.0001 | 0.0046 |
| trec-covid | bm25 | minilm-base | +0.105 | [+0.022, +0.190] | 0.0103 | 0.0618 |
| trec-covid | bge-small | minilm-base | +0.284 | [+0.212, +0.359] | 0.0001 | 0.0046 |
| trec-covid | hm-3fold | minilm-base | -0.253 | [-0.334, -0.174] | 0.0001 | 0.0046 |
