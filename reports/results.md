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

### Within-year task (bottom / middle / top third of each year; chance = 0.333)

| model                             | inputs                             | train   | cv     | test   | test_weighted_f1   | train-test gap   |
|:----------------------------------|:-----------------------------------|:--------|:-------|:-------|:-------------------|:-----------------|
| ordinal (full)                    | qwen3-0.6b_title,specter2_abstract | 0.6738  | 0.4596 | 0.4676 | 0.4683             | 0.2062           |
| ordinal (base)                    | qwen3-0.6b_title                   | 0.6776  | 0.4529 | 0.4571 | 0.4583             | 0.2205           |
| lgbm (full)                       | qwen3-0.6b_title,specter2_abstract | 0.776   | 0.4419 | 0.4492 | 0.4506             | 0.3268           |
| svm (full)                        | qwen3-0.6b_title,specter2_abstract | 0.5149  | 0.4383 | 0.4424 | 0.4437             | 0.0725           |
| lgbm (base)                       | qwen3-0.6b_title                   | 0.707   | 0.4398 | 0.4423 | 0.4441             | 0.2647           |
| mlp (full)                        | qwen3-0.6b_title,specter2_abstract | 0.5406  | 0.439  | 0.4416 | 0.4426             | 0.099            |
| svm (base)                        | qwen3-0.6b_title                   | 0.4929  | 0.4337 | 0.4393 | 0.4407             | 0.0536           |
| mlp (base)                        | qwen3-0.6b_title                   | 0.5152  | 0.4349 | 0.4293 | 0.4323             | 0.0859           |
| stack (full)                      | qwen3-0.6b_title,specter2_abstract | 0.5813  | 0.4164 | 0.4224 | 0.4266             | 0.1589           |
| stack (base)                      | qwen3-0.6b_title                   | 0.5484  | 0.4045 | 0.4022 | 0.4073             | 0.1462           |
| Stanford RF (default, deep trees) | baseline                           | 1.0     | –      | 0.3935 | 0.3958             | 0.6065           |
| Stanford NN (MLP 256-128)         | baseline                           | 0.3685  | –      | 0.3509 | 0.356              | 0.0176           |
| Stanford NB (MultinomialNB)       | baseline                           | 0.2698  | –      | 0.275  | 0.2823             | -0.0052          |
| year-only majority rule           | baseline                           | 0.1721  | –      | 0.1731 | 0.1822             | -0.001           |
| lgbm (base)                       | specter2_title                     | –       | 0.434  | –      | –                  | –                |
| mlp (base)                        | specter2_title                     | –       | 0.4291 | –      | –                  | –                |
| ordinal (base)                    | specter2_title                     | –       | 0.4485 | –      | –                  | –                |
| stack (base)                      | specter2_title                     | –       | 0.3921 | –      | –                  | –                |

### SVM kernel / transformation comparison (best CV macro F1 per variant)

`rbf_exact` was trained on a 30k subsample (its cost grows with n^2); all others on all training rows.

| variant       | base features, orig task   | base features, yn task   | full features, orig task   | full features, yn task   |
|:--------------|:---------------------------|:-------------------------|:---------------------------|:-------------------------|
| linear        | 0.7079 (train 0.731)       | 0.4245 (train 0.478)     | 0.7076 (train 0.745)       | 0.4333 (train 0.514)     |
| rbf_exact     | 0.6972 (train 0.872)       | 0.4306 (train 0.794)     | 0.6931 (train 0.885)       | 0.4314 (train 0.792)     |
| rbf_nystroem  | 0.7063 (train 0.737)       | 0.4303 (train 0.495)     | 0.706 (train 0.748)        | 0.4372 (train 0.522)     |
| rbf_fourier   | 0.6847 (train 0.719)       | 0.4225 (train 0.487)     | 0.6565 (train 0.705)       | 0.429 (train 0.508)      |
| poly_nystroem | 0.707 (train 0.737)        | 0.4286 (train 0.494)     | 0.7067 (train 0.75)        | 0.4363 (train 0.524)     |
| poly_sketch   | 0.409 (train 0.455)        | 0.3741 (train 0.426)     | 0.4122 (train 0.457)       | 0.3849 (train 0.436)     |
