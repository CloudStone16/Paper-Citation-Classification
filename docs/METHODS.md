# Methods, explained step by step

This document explains what the pipeline does and why, assuming little prior ML background.

## 1. The task

Each paper must be put into one of three citation bins: **<60**, **60–120** or **>120** citations
(the bins used by the Stanford CS229 report we compare against). This is *classification*: the
model outputs one of a fixed set of labels.

**F1 score.** For one class, precision = "of the papers I put in this bin, how many belong there",
recall = "of the papers that belong in this bin, how many did I find". F1 is their harmonic mean.
*Macro F1* averages the three per-class F1s equally; *weighted F1* weights them by class size.
Stanford did not say which they used, so we report both.

## 2. The data

The Stanford team used the dataset of Letchford, Moat & Preis (2015): the 20,000 most-cited
papers of each year 2007–2013 (140,000 papers), with title, journal, year and citation count
(counted in late 2014). It is public (CC0) on Zenodo.

**Important property we found.** Because the sample is "top 20,000 *per year*", older papers
have had more time to collect citations, and the year alone decides most of the bin: no 2007–2010
paper has fewer than 60 citations, while 95% of 2013 papers do. A rule that only looks at the year
reaches macro F1 ≈ 0.70 — about the same as Stanford's best model (0.7026).

So we report two tasks:
- **Original task** – the bins above, to compare with Stanford.
- **Within-year task** – each year's papers split into bottom / middle / top third by citations.
  Year gives no information here (guessing = 0.333), so this measures genuine predictive skill.

## 3. Cleaning (`02_clean.py`)

- Fixed broken characters and stripped HTML/LaTeX from titles; normalized journal spellings.
- Removed exact duplicates and placeholder titles.
- Kept the same title appearing in several journals in the same year (e.g. medical guidelines
  co-published in multiple journals – genuinely separate records) but gave them a shared group id.

## 4. Train / test split and cross-validation (`05_split.py`)

- **Test set (15%)** is locked away and used once, at the very end. Using it while choosing models
  would make the final score optimistic.
- **5-fold cross-validation (CV)** on the remaining 85%: the training data is cut into 5 parts;
  each part is predicted by a model trained on the other 4. The average score is our estimate for
  choosing between models.
- **Stratified** = every split keeps the same bin proportions. **Grouped** = copies of the same
  title always land on the same side, so the model cannot memorize a title and see it again.

## 5. Extra features (enrichment, `03`/`04`/`08`)

Stanford concluded their models were limited by *bias*: the inputs (title, journal, year) did not
contain enough information. We add more information about each paper:

1. **Crossref** – searched each title to find the paper's DOI (unique ID); accepted only close
   title + journal matches (≈96% matched).
2. **OpenAlex** – looked up every DOI to get the abstract, authors and their countries,
   institutions, number of references, research field, funding and open-access status.

**Data leakage rule.** Only information that existed when the paper was published is used.
Current citation counts, author h-index and similar values are excluded, because they partly
*contain* the answer.

Features include: number of authors / institutions / countries, international collaboration,
**primary country** (most common author country; ties go to the first author), number of
references, page count, funders, institution types (university, company, government, hospital…),
field and topic, open-access status, abstract length and readability.

## 6. Turning features into numbers the model can use

- **Embeddings.** A pretrained neural network turns text into a vector of numbers whose geometry
  reflects meaning. We use **SPECTER2** (trained on scientific papers) and **Qwen3-Embedding**
  (recent general-purpose model) for titles and abstracts, on the Mac's GPU.
- **PCA** compresses each embedding to 128 numbers while keeping most of the information
  (fitted on training rows only).
- **log1p** compresses heavy-tailed counts (e.g. references: most papers ~30, a few 500).
- **Standardization** rescales each numeric feature to mean 0, spread 1, so no feature dominates
  distance-based models such as SVMs.
- **One-hot encoding** turns a small category (field, year) into yes/no columns.
- **Target encoding** replaces a large category (journal, country, topic) by the average label of
  that category, computed with internal cross-fitting so a paper never encodes its own label.
- All of these are fitted on training data only, inside each CV fold.

## 7. Models

**SVM (support vector machine)** finds the boundary between classes with the widest margin.
A *kernel* lets it draw curved boundaries by implicitly mapping data into a higher-dimensional space.
We compared:

| Variant | Idea |
|---|---|
| Linear | straight boundaries on the features as they are |
| Exact RBF | curved boundaries based on distance; cost grows with n², so trained on 30k rows |
| Nyström RBF | approximates the RBF kernel with 2,000 landmark points, so all rows can be used |
| Random Fourier features | another RBF approximation, using random cosine features |
| Polynomial (Nyström) | boundaries built from pairwise feature products |
| Polynomial sketch | cheap explicit pairwise feature interactions |

We also compared input scalings (standard, quantile, power transform) and class weighting.

**Other models**
- **LightGBM** – many small decision trees built one after another, each fixing the previous ones'
  mistakes. Regularized (limited leaves, row/column subsampling, early stopping) to avoid the
  random forest's overfitting.
- **Ordinal model** – the bins are ordered, so we predict a continuous citation score and cut it
  into bins with thresholds tuned on CV predictions.
- **MLP** – a neural network with two hidden layers, L2 penalty and early stopping.
- **Stacking** – a logistic regression that learns how best to combine the other models' CV outputs.

**Overfitting check.** For every model we compare training, CV and test scores. A large
train–test gap (like Stanford's random forest: 0.9999 vs 0.6336) means the model memorized
the training data.
