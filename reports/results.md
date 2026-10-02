## Results

Macro F1 unless noted. `cv` = mean over the 5 training folds; `test` = held-out 15%, used once.

### Original task (bins <60 / 60-120 / >120)

| Reported by Stanford | Train F1 | Test F1 |
|---|---|---|
| Stanford NB (reported) | 0.4796 | 0.4783 |
| Stanford RF (reported) | 0.9999 | 0.6336 |
| Stanford NN (reported) | 0.7027 | 0.7026 |

| model                             | notes                                                 |    train |       cv |     test |   test_weighted_f1 |   train-test gap |
|:----------------------------------|:------------------------------------------------------|---------:|---------:|---------:|-------------------:|-----------------:|
| Stanford NN (MLP 256-128)         | MiniLM title emb + year + title length + journal code |   0.7134 | nan      |   0.7095 |             0.702  |           0.0039 |
| Stanford RF (default, deep trees) | MiniLM title emb + year + title length + journal code |   1      | nan      |   0.6546 |             0.6513 |           0.3454 |
| Stanford NB (MultinomialNB)       | MiniLM title emb + year + title length + journal code |   0.3544 | nan      |   0.3557 |             0.3749 |          -0.0013 |
| lgbm (base)                       | qwen3-0.6b_title train(fold avg) macroF1=0.8764       | nan      |   0.7114 | nan      |           nan      |         nan      |
| lgbm (base)                       | specter2_title train(fold avg) macroF1=0.8703         | nan      |   0.71   | nan      |           nan      |         nan      |
| mlp (base)                        | qwen3-0.6b_title train(fold avg) macroF1=0.7715       | nan      |   0.7036 | nan      |           nan      |         nan      |
| mlp (base)                        | specter2_title train(fold avg) macroF1=0.7558         | nan      |   0.7019 | nan      |           nan      |         nan      |
| ordinal (base)                    | qwen3-0.6b_title                                      | nan      |   0.7234 | nan      |           nan      |         nan      |
| ordinal (base)                    | specter2_title                                        | nan      |   0.7206 | nan      |           nan      |         nan      |
| stack (base)                      | qwen3-0.6b_title                                      | nan      |   0.7224 | nan      |           nan      |         nan      |
| stack (base)                      | specter2_title                                        | nan      |   0.7187 | nan      |           nan      |         nan      |
| svm (base)                        | qwen3-0.6b_title                                      | nan      |   0.708  | nan      |           nan      |         nan      |

### Within-year task (bottom / middle / top third of each year; chance = 0.333)

| model                             | notes                                                 |    train |       cv |     test |   test_weighted_f1 |   train-test gap |
|:----------------------------------|:------------------------------------------------------|---------:|---------:|---------:|-------------------:|-----------------:|
| Stanford RF (default, deep trees) | MiniLM title emb + year + title length + journal code |   1      | nan      |   0.3935 |             0.3958 |           0.6065 |
| Stanford NN (MLP 256-128)         | MiniLM title emb + year + title length + journal code |   0.3685 | nan      |   0.3509 |             0.356  |           0.0176 |
| Stanford NB (MultinomialNB)       | MiniLM title emb + year + title length + journal code |   0.2698 | nan      |   0.275  |             0.2823 |          -0.0052 |
| lgbm (base)                       | specter2_title train(fold avg) macroF1=0.7478         | nan      |   0.434  | nan      |           nan      |         nan      |
| mlp (base)                        | specter2_title train(fold avg) macroF1=0.5340         | nan      |   0.4291 | nan      |           nan      |         nan      |
| ordinal (base)                    | specter2_title                                        | nan      |   0.4485 | nan      |           nan      |         nan      |
| stack (base)                      | specter2_title                                        | nan      |   0.3921 | nan      |           nan      |         nan      |
