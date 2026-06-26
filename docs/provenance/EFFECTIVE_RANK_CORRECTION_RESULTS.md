# Corrected real GPT-2 effective-rank results

Effective-rank definition: `p_i = sigma_i^2 / sum_j sigma_j^2`.

All three strategies were rerun together for every seed; no old control measurements were reused.

Each strategy used a fresh, identically seeded training loader, so mini-batch order was matched within each seed.

## gpt2_cattn_cfc_5seed

| strategy           |   n |   mean_initial_val_loss |   mean_final_val_loss |   sem_final_val_loss |   mean_val_loss_delta |   sem_val_loss_delta |   mean_perplexity |   sem_perplexity |   mean_trainable_params |   min_rank |   mean_rank |   max_rank |   wins |
|:-------------------|----:|------------------------:|----------------------:|---------------------:|----------------------:|---------------------:|------------------:|-----------------:|------------------------:|-----------:|------------:|-----------:|-------:|
| spectral_effective |   5 |                 3.94864 |               3.45726 |           0.00159468 |             -0.491373 |           0.00159468 |           31.7302 |        0.0506301 |                  331776 |          1 |     3.70833 |       10.6 |      4 |
| uniform_r4         |   5 |                 3.94864 |               3.45875 |           0.00168992 |             -0.489889 |           0.00168992 |           31.7773 |        0.0537486 |                  331776 |          4 |     4       |        4   |      1 |
| gradient_norm      |   5 |                 3.94864 |               3.50472 |           0.00139032 |             -0.443913 |           0.00139032 |           33.2724 |        0.0462428 |                  330086 |          1 |     4.25833 |       16   |      0 |

Pairwise spectral differences:

| comparison                   |   final_val_loss_diff |   perplexity_diff |
|:-----------------------------|----------------------:|------------------:|
| spectral_minus_gradient_norm |           -0.0474607  |        -1.54221   |
| spectral_minus_uniform_r4    |           -0.00148455 |        -0.0471599 |

## gpt2_attnproj_3seed

| strategy           |   n |   mean_initial_val_loss |   mean_final_val_loss |   sem_final_val_loss |   mean_val_loss_delta |   sem_val_loss_delta |   mean_perplexity |   sem_perplexity |   mean_trainable_params |   min_rank |   mean_rank |   max_rank |   wins |
|:-------------------|----:|------------------------:|----------------------:|---------------------:|----------------------:|---------------------:|------------------:|-----------------:|------------------------:|-----------:|------------:|-----------:|-------:|
| uniform_r4         |   3 |                 3.94864 |               3.44794 |          0.000559587 |             -0.500695 |          0.000559587 |           31.4356 |        0.0175913 |                  405504 |          4 |     4       |          4 |      3 |
| spectral_effective |   3 |                 3.94864 |               3.453   |          0.00260351  |             -0.495635 |          0.00260351  |           31.5953 |        0.0823134 |                  404992 |          1 |     5.12963 |         16 |      0 |
| gradient_norm      |   3 |                 3.94864 |               3.48469 |          0.00138658  |             -0.463945 |          0.00138658  |           32.6124 |        0.0452489 |                  405248 |          1 |     4.47222 |         16 |      0 |

Pairwise spectral differences:

| comparison                   |   final_val_loss_diff |   perplexity_diff |
|:-----------------------------|----------------------:|------------------:|
| spectral_minus_gradient_norm |           -0.0316901  |         -1.01713  |
| spectral_minus_uniform_r4    |            0.00506008 |          0.159674 |

## Paper follow-up

The generated LaTeX table rows and all hard-coded numerical claims in `Paper/paper.tex` have been reconciled with these corrected values. The rebuilt top-level `paper.pdf` uses the corrected table and cautious descriptive wording.
