## Results

See [`docs/RESULTS_SUMMARY.md`](docs/RESULTS_SUMMARY.md) for a plain-language summary and [`docs/METHODS.md`](docs/METHODS.md) for how everything works.

Macro F1 unless noted. `cv` = mean over the 5 training folds; `test` = held-out 15%, used once.

### Original task (bins <60 / 60-120 / >120)

| Reported by Stanford | Train F1 | Test F1 |
|---|---|---|
| Stanford NB (reported) | 0.4796 | 0.4783 |
| Stanford RF (reported) | 0.9999 | 0.6336 |
| Stanford NN (reported) | 0.7027 | 0.7026 |

| model                             | inputs                             | train   | cv     | test   | test_weighted_f1   | train-test gap   |
|:----------------------------------|:-----------------------------------|:--------|:-------|:-------|:-------------------|:-----------------|
| ordinal (full)                    | qwen3-0.6b_title,specter2_abstract | 0.8546  | 0.7273 | 0.7313 | 0.7233             | 0.1233           |
| stack (full)                      | qwen3-0.6b_title,specter2_abstract | 0.8062  | 0.7267 | 0.7299 | 0.7226             | 0.0763           |
| stack (base)                      | qwen3-0.6b_title                   | 0.8027  | 0.7223 | 0.7272 | 0.7197             | 0.0755           |
| ordinal (base)                    | qwen3-0.6b_title                   | 0.8446  | 0.7235 | 0.7268 | 0.7188             | 0.1178           |
| lgbm (full)                       | qwen3-0.6b_title,specter2_abstract | 0.9198  | 0.7112 | 0.7161 | 0.708              | 0.2037           |
| lgbm (base)                       | qwen3-0.6b_title                   | 0.8585  | 0.712  | 0.7159 | 0.7078             | 0.1426           |
| svm (full)                        | qwen3-0.6b_title,specter2_abstract | 0.7419  | 0.7076 | 0.7116 | 0.7039             | 0.0303           |
| svm (base)                        | qwen3-0.6b_title                   | 0.7291  | 0.708  | 0.7115 | 0.7043             | 0.0176           |
| Stanford NN (MLP 256-128)         | baseline                           | 0.7134  | –      | 0.7095 | 0.702              | 0.0039           |
| mlp (base)                        | qwen3-0.6b_title                   | 0.7465  | 0.7038 | 0.7084 | 0.6993             | 0.0381           |
| mlp (full)                        | qwen3-0.6b_title,specter2_abstract | 0.7608  | 0.7009 | 0.7024 | 0.6948             | 0.0584           |
| year-only majority rule           | baseline                           | 0.6968  | –      | 0.6998 | 0.6923             | -0.003           |
| Stanford RF (default, deep trees) | baseline                           | 1.0     | –      | 0.6546 | 0.6513             | 0.3454           |
| Stanford NB (MultinomialNB)       | baseline                           | 0.3544  | –      | 0.3557 | 0.3749             | -0.0013          |
| lgbm (base)                       | specter2_title                     | –       | 0.71   | –      | –                  | –                |
| mlp (base)                        | specter2_title                     | –       | 0.7019 | –      | –                  | –                |
| ordinal (base)                    | specter2_title                     | –       | 0.7206 | –      | –                  | –                |
| stack (base)                      | specter2_title                     | –       | 0.7187 | –      | –                  | –                |

### SVM kernel / transformation comparison (best CV macro F1 per variant)

`rbf_exact` was trained on a 30k subsample (its cost grows with n^2); all others on all training rows.

| variant       | base features        | full features        |
|:--------------|:---------------------|:---------------------|
| linear        | 0.7079 (train 0.731) | 0.7076 (train 0.745) |
| rbf_exact     | 0.6972 (train 0.872) | 0.6931 (train 0.885) |
| rbf_nystroem  | 0.7063 (train 0.737) | 0.706 (train 0.748)  |
| rbf_fourier   | 0.6847 (train 0.719) | 0.6565 (train 0.705) |
| poly_nystroem | 0.707 (train 0.737)  | 0.7067 (train 0.75)  |
| poly_sketch   | 0.409 (train 0.455)  | 0.4122 (train 0.457) |
