# Results summary

All scores are **macro F1** on the held-out test set (15% of papers, used once at the very end),
unless marked CV (5-fold cross-validation on the training set). Full tables: [`reports/results.md`](../reports/results.md).

## Headline

**Best model: ordinal LightGBM on the full (enriched) feature set**, chosen by its CV score
(the highest of all models), then evaluated once on the test set.

Task: Stanford's citation bins (<60 / 60–120 / >120).

| Model | Test macro F1 | Test weighted F1 |
|---|---|---|
| **Ours: ordinal model, full features** | **0.7313** | **0.7233** |
| Ours: stacking ensemble, full features | 0.7299 | 0.7226 |
| Ours: ordinal model, base features (no enrichment) | 0.7268 | 0.7188 |
| Ours: best SVM (linear), full features | 0.7116 | 0.7039 |
| Stanford NN, as reported in their paper | 0.7026 (metric not stated) | |
| Stanford NN, rebuilt by us on the same split | 0.7095 | 0.7020 |
| Year-only rule (predict the most common bin for the paper's year) | 0.6998 | 0.6923 |

**+0.029 over Stanford's reported 0.7026**, +0.022 over their method rebuilt on our split.
The gain is modest because the year alone already scores 0.70 on this task: the dataset holds the
top 20,000 most-cited papers *per year*, so older papers are almost automatically in higher bins.

## Overfitting check

| Model | Train | CV | Test | Comment |
|---|---|---|---|---|
| Stanford-style RF (rebuilt) | 1.000 | – | 0.655 | memorizes the training set |
| Ours: ordinal, full | 0.855 | 0.727 | 0.731 | train > test, but CV ≈ test, so the score is reliable |
| Ours: linear SVM, full | 0.742 | 0.708 | 0.712 | smallest gap of all models |

CV and test agree to within ~0.005 for every model, so the reported test scores are trustworthy.

## What helped (and what did not)

- **Treating the bins as ordered** (predict a citation score, then cut it into bins) was the
  single best modelling choice: about +0.015 over a plain LightGBM classifier on the same features.
- **Encoding journals by their average citation level** (target encoding) instead of an integer code.
- **Extra OpenAlex features** (abstract, authors, countries, references, field, funding, open access):
  about +0.005 (ordinal model: 0.7268 → 0.7313). Smaller than hoped: within this
  "top cited papers only" dataset these properties barely differ between papers (every individual
  feature correlates only weakly with citations, |ρ| ≤ 0.06, journal size aside).
- **SVM kernels**: on these features, curved boundaries did not beat a straight one. Linear,
  Nyström RBF and Nyström polynomial all scored ≈0.706–0.708 CV. Exact RBF overfit
  (train 0.87–1.00), random Fourier features and the polynomial sketch were worse.
  Scaler choice (standard / quantile / power) made no difference.
- **Embedding model**: Qwen3-0.6B vs SPECTER2 for titles changed scores by ~0.003.
- **Did not help**: a journal × year feature (overfit, −0.02 CV) and class weighting.


