| dataset (n queries) | system | nDCG@10 [95% CI] |
|---|---|---|
| scifact (300) | base | 0.645 [0.600, 0.689] |
| scifact (300) | published 3-fold | 0.249 [0.209, 0.290] |
| scifact (300) | recipe 1 (earlier), averaged | 0.580 [0.536, 0.626] |
| scifact (300) | recipe v2, averaged | 0.639 [0.594, 0.684] |
| scifact (300) | BGE-small | 0.713 [0.669, 0.755] |
| nfcorpus (323) | base | 0.316 [0.282, 0.350] |
| nfcorpus (323) | published 3-fold | 0.069 [0.054, 0.085] |
| nfcorpus (323) | recipe 1 (earlier), averaged | 0.293 [0.260, 0.327] |
| nfcorpus (323) | recipe v2, averaged | 0.311 [0.278, 0.345] |
| nfcorpus (323) | BGE-small | 0.343 [0.308, 0.379] |
| trec-covid (50) | base | 0.472 [0.389, 0.558] |
| trec-covid (50) | published 3-fold | 0.219 [0.158, 0.286] |
| trec-covid (50) | recipe 1 (earlier), averaged | 0.443 [0.355, 0.534] |
| trec-covid (50) | recipe v2, averaged | 0.455 [0.371, 0.542] |
| trec-covid (50) | BGE-small | 0.756 [0.698, 0.813] |
| medquad (4599) | base | 0.524 [0.512, 0.535] |
| medquad (4599) | published 3-fold | 0.180 [0.171, 0.189] |
| medquad (4599) | recipe 1 (earlier), averaged | 0.419 [0.409, 0.429] |
| medquad (4599) | recipe v2, averaged | 0.501 [0.490, 0.512] |
| medquad (4599) | BGE-small | 0.587 [0.576, 0.598] |

| dataset | v2 avg minus base [95% CI] | Holm p | earlier recipe minus base [95% CI] |
|---|---|---|---|
| scifact | -0.006 [-0.019, +0.008] | 0.3937 | -0.065 [-0.092, -0.036] |
| nfcorpus | -0.005 [-0.012, +0.002] | 0.3573 | -0.023 [-0.035, -0.011] |
| trec-covid | -0.017 [-0.038, +0.004] | 0.3573 | -0.029 [-0.067, +0.007] |
| medquad | -0.023 [-0.028, -0.017] | 0.0004 | -0.105 [-0.113, -0.097] |
