# Paper Citation Classification

Predict which citation bin (`<60`, `60–120`, `>120`) an academic paper falls into, improving on the
Stanford CS229 project in [`docs/26647373.pdf`](docs/26647373.pdf) (best test F1 0.7026, neural network)
without overfitting.

## Data

Letchford, Moat & Preis (2015), *The advantage of short paper titles* — ~140,000 papers from 2007–2013
(the 20,000 most-cited papers per year, citation counts from Scopus, Oct–Nov 2014).
Zenodo record [4989552](https://zenodo.org/records/4989552), DOI `10.5061/dryad.hg3j0`, licence CC0.

The raw CSV is not committed; `src/01_download.py` fetches it and checks its MD5.
Each paper is enriched with metadata from Crossref (DOI lookup) and OpenAlex (abstract, authors,
countries, references, field, funding), using only information available at publication time.

## Pipeline

| Step | Script | Output |
|---|---|---|
| Download | `src/01_download.py` | `data/raw/LMP2015.csv` |
| Clean + label | `src/02_clean.py` | `data/interim/clean.parquet` |
| Crossref DOI match | `src/03_crossref_match.py` | `data/interim/crossref_cache.jsonl` |
| OpenAlex enrichment | `src/04_openalex_enrich.py` | `data/interim/openalex.parquet` |
| Features + split | `src/05_features.py` | `data/processed/` |
| Embeddings | `src/06_embed.py` | `data/processed/*.npy` |
| SVM transform comparison | `src/07_svm_transforms.py` | results table |
| MLP / LightGBM / fine-tune / stacking | `src/08_models.py` | results table |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in OPENALEX_API_KEY and CONTACT_EMAIL
```

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
