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

| Model | Train F1 | CV F1 | Test F1 |
|---|---|---|---|
| Stanford NB (reported) | 0.4796 | – | 0.4783 |
| Stanford RF (reported) | 0.9999 | – | 0.6336 |
| Stanford NN (reported) | 0.7027 | – | 0.7026 |
