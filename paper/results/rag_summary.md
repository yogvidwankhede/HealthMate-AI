| generator | retrieval | accuracy [95% CI] | macro-F1 | diff vs none [95% CI] | pred yes/no/maybe | yes-vs-no acc (secondary, exploratory) |
|---|---|---|---|---|---|---|
| base | base | 0.162 [0.130, 0.194] | 0.128 | +0.006 [-0.012, +0.024] | 57/0/443 | 0.615 (n=442) |
| base | hm3fold | 0.150 [0.120, 0.182] | 0.116 | -0.006 [-0.024, +0.014] | 43/1/456 | 0.613 (n=442) |
| base | none | 0.156 [0.124, 0.188] | 0.122 |  | 40/0/460 | 0.620 (n=442) |
| base | ours | 0.160 [0.128, 0.192] | 0.125 | +0.004 [-0.016, +0.024] | 51/0/449 | 0.620 (n=442) |

Majority-class ('yes') accuracy: 0.552

Invalid runs (all logits NaN; adapter weights are NaN, see adapter_nan.json), excluded: lora123/base, lora123/none, lora42/base, lora42/hm3fold, lora42/none, lora42/ours, lora999/base, lora999/none
