| Setting | Strategy | Final validation loss (SEM) | Loss Δ vs uniform | Bootstrap 95% CI | Wins |
| --- | --- | --- | --- | --- | --- |
| c_attn, c_fc | spectral effective | 3.4513 (0.0010) | -0.0070 | [-0.0083, -0.0060] | 8/8 |
| c_attn, c_fc | EVA-style allocation | 3.4537 (0.0012) | -0.0046 | [-0.0049, -0.0043] | 8/8 |
| c_attn, c_fc | uniform rank 4 | 3.4583 (0.0012) | 0 (reference) | — | — |
| c_attn, c_fc | GoRA-style allocation | 3.4610 (0.0011) | +0.0027 | [+0.0022, +0.0032] | 0/8 |
| c_attn, c_fc | gradient norm | 3.4857 (0.0011) | +0.0275 | [+0.0262, +0.0285] | 0/8 |
| c_attn, c_fc | FIM-LoRA-style allocation | 3.4996 (0.0015) | +0.0413 | [+0.0385, +0.0445] | 0/8 |
| + attn.c_proj | spectral effective | 3.4455 (0.0011) | +0.0011 | [+0.0009, +0.0013] | 0/3 |
| + attn.c_proj | EVA-style allocation | 3.4506 (0.0013) | +0.0062 | [+0.0061, +0.0063] | 0/3 |
| + attn.c_proj | uniform rank 4 | 3.4444 (0.0012) | 0 (reference) | — | — |
| + attn.c_proj | GoRA-style allocation | 3.4655 (0.0012) | +0.0211 | [+0.0205, +0.0218] | 0/3 |
| + attn.c_proj | gradient norm | 3.4693 (0.0020) | +0.0249 | [+0.0238, +0.0266] | 0/3 |
| + attn.c_proj | FIM-LoRA-style allocation | 3.4783 (0.0027) | +0.0339 | [+0.0310, +0.0367] | 0/3 |
