## Results

Macro F1 unless noted. `cv` = mean over the 5 training folds; `test` = held-out 15%, used once.

### Original task (bins <60 / 60-120 / >120)

| Reported by Stanford | Train F1 | Test F1 |
|---|---|---|
| Stanford NB (reported) | 0.4796 | 0.4783 |
| Stanford RF (reported) | 0.9999 | 0.6336 |
| Stanford NN (reported) | 0.7027 | 0.7026 |

| model                             | notes                                                        |    train |       cv |     test |   test_weighted_f1 |   train-test gap |
|:----------------------------------|:-------------------------------------------------------------|---------:|---------:|---------:|-------------------:|-----------------:|
| ordinal (full)                    | qwen3-0.6b_title,specter2_abstract                           |   0.8546 | nan      |   0.7313 |             0.7233 |           0.1233 |
| stack (full)                      | qwen3-0.6b_title,specter2_abstract                           |   0.8062 | nan      |   0.7299 |             0.7226 |           0.0763 |
| stack (base)                      | qwen3-0.6b_title                                             |   0.8027 | nan      |   0.7272 |             0.7197 |           0.0755 |
| ordinal (base)                    | qwen3-0.6b_title                                             |   0.8446 | nan      |   0.7268 |             0.7188 |           0.1178 |
| lgbm (full)                       | qwen3-0.6b_title,specter2_abstract                           |   0.9198 | nan      |   0.7161 |             0.708  |           0.2037 |
| lgbm (base)                       | qwen3-0.6b_title                                             |   0.8585 | nan      |   0.7159 |             0.7078 |           0.1426 |
| svm (full)                        | qwen3-0.6b_title,specter2_abstract                           |   0.7419 | nan      |   0.7116 |             0.7039 |           0.0303 |
| svm (base)                        | qwen3-0.6b_title                                             |   0.7291 | nan      |   0.7115 |             0.7043 |           0.0176 |
| Stanford NN (MLP 256-128)         | MiniLM title emb + year + title length + journal code        |   0.7134 | nan      |   0.7095 |             0.702  |           0.0039 |
| mlp (base)                        | qwen3-0.6b_title                                             |   0.7465 | nan      |   0.7084 |             0.6993 |           0.0381 |
| mlp (full)                        | qwen3-0.6b_title,specter2_abstract                           |   0.7608 | nan      |   0.7024 |             0.6948 |           0.0584 |
| Stanford RF (default, deep trees) | MiniLM title emb + year + title length + journal code        |   1      | nan      |   0.6546 |             0.6513 |           0.3454 |
| Stanford NB (MultinomialNB)       | MiniLM title emb + year + title length + journal code        |   0.3544 | nan      |   0.3557 |             0.3749 |          -0.0013 |
| lgbm (base)                       | qwen3-0.6b_title train(fold avg) macroF1=0.8764              | nan      |   0.7114 | nan      |           nan      |         nan      |
| lgbm (base)                       | qwen3-0.6b_title train(fold avg) macroF1=0.8862              | nan      |   0.712  | nan      |           nan      |         nan      |
| lgbm (base)                       | specter2_title train(fold avg) macroF1=0.8703                | nan      |   0.71   | nan      |           nan      |         nan      |
| lgbm (full)                       | qwen3-0.6b_title,specter2_abstract train(fold avg) macroF1=0 | nan      |   0.7085 | nan      |           nan      |         nan      |
| lgbm (full)                       | qwen3-0.6b_title,specter2_abstract train(fold avg) macroF1=0 | nan      |   0.7112 | nan      |           nan      |         nan      |
| mlp (base)                        | qwen3-0.6b_title train(fold avg) macroF1=0.7715              | nan      |   0.7036 | nan      |           nan      |         nan      |
| mlp (base)                        | qwen3-0.6b_title train(fold avg) macroF1=0.7716              | nan      |   0.7038 | nan      |           nan      |         nan      |
| mlp (base)                        | specter2_title train(fold avg) macroF1=0.7558                | nan      |   0.7019 | nan      |           nan      |         nan      |
| mlp (full)                        | qwen3-0.6b_title,specter2_abstract train(fold avg) macroF1=0 | nan      |   0.7009 | nan      |           nan      |         nan      |
| mlp (full)                        | qwen3-0.6b_title,specter2_abstract train(fold avg) macroF1=0 | nan      |   0.6946 | nan      |           nan      |         nan      |
| ordinal (base)                    | qwen3-0.6b_title                                             | nan      |   0.7235 | nan      |           nan      |         nan      |
| ordinal (base)                    | specter2_title                                               | nan      |   0.7206 | nan      |           nan      |         nan      |
| ordinal (full)                    | qwen3-0.6b_title,specter2_abstract                           | nan      |   0.7273 | nan      |           nan      |         nan      |
| stack (base)                      | qwen3-0.6b_title                                             | nan      |   0.7223 | nan      |           nan      |         nan      |
| stack (base)                      | specter2_title                                               | nan      |   0.7187 | nan      |           nan      |         nan      |
| stack (full)                      | qwen3-0.6b_title,specter2_abstract                           | nan      |   0.7267 | nan      |           nan      |         nan      |
| svm (base)                        | qwen3-0.6b_title                                             | nan      |   0.708  | nan      |           nan      |         nan      |
| svm (full)                        | qwen3-0.6b_title,specter2_abstract                           | nan      |   0.7076 | nan      |           nan      |         nan      |

### Within-year task (bottom / middle / top third of each year; chance = 0.333)

| model                             | notes                                                        |    train |       cv |     test |   test_weighted_f1 |   train-test gap |
|:----------------------------------|:-------------------------------------------------------------|---------:|---------:|---------:|-------------------:|-----------------:|
| ordinal (full)                    | qwen3-0.6b_title,specter2_abstract                           |   0.6738 | nan      |   0.4676 |             0.4683 |           0.2062 |
| ordinal (base)                    | qwen3-0.6b_title                                             |   0.6776 | nan      |   0.4571 |             0.4583 |           0.2205 |
| lgbm (full)                       | qwen3-0.6b_title,specter2_abstract                           |   0.776  | nan      |   0.4492 |             0.4506 |           0.3268 |
| svm (full)                        | qwen3-0.6b_title,specter2_abstract                           |   0.5149 | nan      |   0.4424 |             0.4437 |           0.0725 |
| lgbm (base)                       | qwen3-0.6b_title                                             |   0.707  | nan      |   0.4423 |             0.4441 |           0.2647 |
| mlp (full)                        | qwen3-0.6b_title,specter2_abstract                           |   0.5406 | nan      |   0.4416 |             0.4426 |           0.099  |
| svm (base)                        | qwen3-0.6b_title                                             |   0.4929 | nan      |   0.4393 |             0.4407 |           0.0536 |
| mlp (base)                        | qwen3-0.6b_title                                             |   0.5152 | nan      |   0.4293 |             0.4323 |           0.0859 |
| stack (full)                      | qwen3-0.6b_title,specter2_abstract                           |   0.5813 | nan      |   0.4224 |             0.4266 |           0.1589 |
| stack (base)                      | qwen3-0.6b_title                                             |   0.5484 | nan      |   0.4022 |             0.4073 |           0.1462 |
| Stanford RF (default, deep trees) | MiniLM title emb + year + title length + journal code        |   1      | nan      |   0.3935 |             0.3958 |           0.6065 |
| Stanford NN (MLP 256-128)         | MiniLM title emb + year + title length + journal code        |   0.3685 | nan      |   0.3509 |             0.356  |           0.0176 |
| Stanford NB (MultinomialNB)       | MiniLM title emb + year + title length + journal code        |   0.2698 | nan      |   0.275  |             0.2823 |          -0.0052 |
| lgbm (base)                       | qwen3-0.6b_title train(fold avg) macroF1=0.7541              | nan      |   0.4398 | nan      |           nan      |         nan      |
| lgbm (base)                       | specter2_title train(fold avg) macroF1=0.7478                | nan      |   0.434  | nan      |           nan      |         nan      |
| lgbm (full)                       | qwen3-0.6b_title,specter2_abstract train(fold avg) macroF1=0 | nan      |   0.4419 | nan      |           nan      |         nan      |
| mlp (base)                        | qwen3-0.6b_title train(fold avg) macroF1=0.5408              | nan      |   0.4349 | nan      |           nan      |         nan      |
| mlp (base)                        | specter2_title train(fold avg) macroF1=0.5340                | nan      |   0.4291 | nan      |           nan      |         nan      |
| mlp (full)                        | qwen3-0.6b_title,specter2_abstract train(fold avg) macroF1=0 | nan      |   0.439  | nan      |           nan      |         nan      |
| ordinal (base)                    | qwen3-0.6b_title                                             | nan      |   0.4529 | nan      |           nan      |         nan      |
| ordinal (base)                    | specter2_title                                               | nan      |   0.4485 | nan      |           nan      |         nan      |
| ordinal (full)                    | qwen3-0.6b_title,specter2_abstract                           | nan      |   0.4596 | nan      |           nan      |         nan      |
| stack (base)                      | qwen3-0.6b_title                                             | nan      |   0.4045 | nan      |           nan      |         nan      |
| stack (base)                      | specter2_title                                               | nan      |   0.3921 | nan      |           nan      |         nan      |
| stack (full)                      | qwen3-0.6b_title,specter2_abstract                           | nan      |   0.4164 | nan      |           nan      |         nan      |
| svm (base)                        | qwen3-0.6b_title                                             | nan      |   0.4337 | nan      |           nan      |         nan      |
| svm (full)                        | qwen3-0.6b_title,specter2_abstract                           | nan      |   0.4383 | nan      |           nan      |         nan      |
