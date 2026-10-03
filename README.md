# Paper Citation Classification

Predict which citation bin (`<60`, `60–120`, `>120`) an academic paper falls into, improving on the
Stanford CS229 project in [`docs/26647373.pdf`](docs/26647373.pdf) (best test F1 0.7026, neural network)
without overfitting.

**Result:** our best model (ordinal LightGBM on enriched features) reaches **test macro F1 0.7313**
(weighted 0.7233), against Stanford's 0.7026 and a year-only rule's 0.6998. Details below and in
[`docs/RESULTS_SUMMARY.md`](docs/RESULTS_SUMMARY.md).

## Data

Letchford, Moat & Preis (2015), *The advantage of short paper titles* — ~140,000 papers from 2007–2013
(the 20,000 most-cited papers per year, citation counts from Scopus, Oct–Nov 2014).
Zenodo record [4989552](https://zenodo.org/records/4989552), DOI `10.5061/dryad.hg3j0`, licence CC0.

Each paper is enriched with metadata from Crossref (DOI lookup by title) and OpenAlex (abstract,
authors, countries, institutions, references, field, funding, open access), using only information
available at publication time. The `data/` and `models/` folders are not committed (too large);
the scripts below recreate them.

## Setup

Requires **Python 3.12** (3.14 is too new for some ML libraries) and macOS or Linux.
On Apple Silicon the embeddings run on the GPU (MPS).

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env    # then fill in CONTACT_EMAIL and OPENALEX_API_KEY (free key from openalex.org)
```

All commands below are run from the project folder.

## Quick demo (uses the saved model)

Requires `models/full_orig.joblib` (created by step 13 below).

```bash
.venv/bin/python src/predict.py --task orig --n 10            # 10 random test papers: actual vs predicted bin
.venv/bin/python src/predict.py --task orig --title graphene  # test papers whose title contains a word
.venv/bin/python src/predict.py --task orig --evaluate        # test scores of all saved models
```

Each command takes a few seconds; nothing is retrained.

## Running the full pipeline

Each step reads the files written by the previous ones, so run them in order.

| # | Command | What it does | Time |
|---|---|---|---|
| 1 | `.venv/bin/python src/01_download.py` | Download the dataset and verify its MD5 | seconds |
| 2 | `.venv/bin/python src/02_clean.py` | Clean titles/journals, remove duplicates, add bin labels | ~1 min |
| 3 | `.venv/bin/python src/03_crossref_match.py crawl` | Search each title on Crossref for its DOI (resumable) | ~8–9 h |
| 4 | `.venv/bin/python src/03_crossref_match.py match` | Accept only close title + journal matches (~96%) | ~1 min |
| 5 | `.venv/bin/python src/04_openalex_enrich.py` | Fetch publication-time metadata for every DOI (resumable) | ~15 min |
| 6 | `.venv/bin/python src/05_split.py` | Fixed 85/15 train/test split + 5 CV folds (stratified, grouped) | seconds |
| 7 | `.venv/bin/python src/06_embed.py --model minilm --text title` | Title embeddings for the Stanford baseline | ~1 min |
| 8 | `.venv/bin/python src/06_embed.py --model qwen3-0.6b --text title` | Title embeddings for our models | ~20 min |
| 9 | `.venv/bin/python src/07_baseline.py` | Rebuild Stanford's NB / RF / NN and the year-only rule | ~2 min |
| 10 | `.venv/bin/python src/08_features.py` | Build the full feature table and extract abstracts | ~3 min |
| 11 | `.venv/bin/python src/06_embed.py --model specter2 --text abstract` | Abstract embeddings | ~25 min |
| 12 | `.venv/bin/python src/09_svm.py --feature-set full --emb qwen3-0.6b_title,specter2_abstract --task orig` | Compare 6 SVM kernels / transformations on CV; save the best setting | ~45–60 min |
| 13 | `.venv/bin/python src/10_models.py --feature-set full --emb qwen3-0.6b_title,specter2_abstract --task orig --final` | Train SVM, LightGBM, ordinal, MLP, stacking; evaluate once on test; save `models/full_orig.joblib` | ~15 min |
| 14 | `.venv/bin/python src/11_report.py` | Rebuild the results tables in this README and `reports/results.md` | seconds |

Use `--feature-set base --emb qwen3-0.6b_title` in steps 12–13 for the comparison without enrichment.

Helpers:
- `src/enrich_loop.sh` – repeats steps 4–5 every 30 min while the crawl runs, stops when enrichment is complete.
- `src/run_final.sh` – runs steps 10–14 in one go once enrichment is complete.
- `./status.sh` – shows which pipeline jobs are running and their progress.

Notes: one heavy job at a time on a 24 GB machine. LightGBM uses 6 threads by default
(`LGBM_THREADS`), which reproduces the reported results exactly; other thread counts can shift
LightGBM-based scores by about ±0.002.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Pipeline scripts (numbered in run order), `common.py` / `config.py` shared helpers, `predict.py` demo |
| `reports/` | `results.csv` (every score logged), `results.md`, `svm_cv.csv`, chosen SVM settings, cleaning report |
| `docs/` | Stanford reference paper, `METHODS.md` (how it works), `RESULTS_SUMMARY.md` (findings) |
| `data/`, `models/` | Created by the pipeline, not committed |

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
