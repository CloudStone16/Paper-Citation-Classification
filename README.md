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
