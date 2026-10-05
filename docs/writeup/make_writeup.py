"""Build the 2-page project write-up: docs/Writeup.pdf

Run from the project folder:  .venv/bin/python docs/writeup/make_writeup.py
Numbers come from reports/results.csv and reports/confusion_matrices.png (src/12_confusion.py).
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "Writeup.pdf"
TEAM = "Anirudh Ramesh and [teammate name]"
REPO = "github.com/CloudStone16/Paper-Citation-Classification (private)"

ss = getSampleStyleSheet()
body = ParagraphStyle("body", parent=ss["Normal"], fontName="Helvetica", fontSize=8.4, leading=10.4,
                      spaceAfter=2.5)
small = ParagraphStyle("small", parent=body, fontSize=7.4, leading=9)
h = ParagraphStyle("h", parent=body, fontName="Helvetica-Bold", fontSize=9.6, leading=11.5,
                   spaceBefore=4, spaceAfter=2, textColor=colors.HexColor("#1f3b63"))
title = ParagraphStyle("title", parent=body, fontName="Helvetica-Bold", fontSize=13.5, leading=16,
                       alignment=TA_CENTER, spaceAfter=1)
sub = ParagraphStyle("sub", parent=body, fontSize=8.4, alignment=TA_CENTER, textColor=colors.HexColor("#444444"))
cell = ParagraphStyle("cell", parent=body, fontSize=7.4, leading=8.8, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Helvetica-Bold")
bullet = ParagraphStyle("bullet", parent=body, leftIndent=10, bulletIndent=2, spaceAfter=2)


def table(rows, widths, bold_rows=(), header=True):
    data = [[Paragraph(str(c), cellb if (r == 0 and header) or r in bold_rows else cell) for c in row]
            for r, row in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], hAlign="LEFT")
    style = [("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b0b7c3")),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
             ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6ecf5")))
    for r in bold_rows:
        style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#eef7ee")))
    t.setStyle(TableStyle(style))
    return t


def P(text, style=body):
    return Paragraph(text, style)


story = [
    P("Paper Citation Classification", title),
    P(f"UE24CS352A Machine Learning – Mini-Project Write-up &nbsp;·&nbsp; Team: {TEAM} &nbsp;·&nbsp; {REPO}", sub),
    Spacer(1, 4),

    P("1. Problem statement", h),
    P("Predict, from information available when a paper is published, which citation bin it will fall into: "
      "<b>&lt;60</b>, <b>60–120</b> or <b>&gt;120</b> citations. The benchmark is the Stanford CS229 project "
      "<i>Paper Citation Classification</i> (Bui, Saavedra &amp; Liu, 2019), which used title, journal and year "
      "with Naive Bayes, a random forest and a neural network. Their best model (neural network) reached "
      "<b>test F1 0.7026</b>; the random forest overfit badly (train 0.9999, test 0.6336), and the authors "
      "concluded the remaining error was <i>bias</i> – the inputs did not carry enough information. Our goal: a "
      "higher F1 without overfitting, using richer features and better models."),

    P("2. Dataset", h),
    P("We use the same dataset as Stanford: Letchford, Moat &amp; Preis (2015), the <b>20,000 most-cited papers of "
      "each year 2007–2013</b> (140,000 papers; title, journal, year, citation count from Scopus, Oct–Nov 2014; "
      "Zenodo, CC0). <b>Cleaning:</b> fixed broken characters and HTML/LaTeX in titles, normalized journal names "
      "(5,278 → 5,075), removed 31 duplicates and 1 placeholder title, leaving <b>139,968 papers</b> "
      "(bins: 29.6% / 39.3% / 31.1%). <b>Split:</b> stratified 85/15 train/test (118,971 / 20,997), grouped so "
      "that the same title co-published in several journals never appears on both sides; 5-fold cross-validation "
      "(CV) on the training set for all model choices; the test set was used once, at the end."),
    P("<b>Key property.</b> Because the sample is the top 20k <i>per year</i>, older papers are automatically in "
      "higher bins (the least-cited 2007 paper has 103 citations; the least-cited 2013 paper has 16). A rule "
      "that predicts the most common bin of the paper's year already scores <b>macro F1 0.700</b> – the bar any "
      "model has to clear."),
    P("<b>Enrichment.</b> Each title was searched on <b>Crossref</b> to find its DOI (accepted only for close "
      "title + journal matches; <b>96.3% matched</b>), then <b>OpenAlex</b> was queried for publication-time "
      "metadata (abstract for 67% of matched papers, author countries for 97%). <b>Leakage rule:</b> current "
      "citation counts, author h-index and similar fields were never requested, since they contain the answer."),

    P("3. Feature set", h),
    table([
        ["Group", "Stanford baseline", "Ours (full feature set)"],
        ["Text", "Title embedding (Universal Sentence Encoder, 256-d; MiniLM 384-d in our rebuild)",
         "Title: Qwen3-Embedding-0.6B (1024-d); abstract: SPECTER2 (768-d, trained on scientific papers); "
         "each compressed to 128 dims with PCA"],
        ["Journal", "Integer code per journal", "Target encoding (journal's average label, cross-fitted) + journal size"],
        ["Year / title", "Year, title length", "Year (numeric + one-hot), title chars / words, colon, question mark, "
         "digits, acronym"],
        ["Authors", "–", "No. of authors, institutions, countries; international-collaboration flag; "
         "primary country (most common author country) and first-author country"],
        ["Paper", "–", "References, pages, keywords, funders, open-access status, document type, field / subfield / "
         "topic, institution-type mix (university, company, government, hospital, …)"],
        ["Abstract", "–", "Has abstract, length, readability (Flesch), words per sentence"],
    ], [2.0, 5.0, 10.9]),
    P("In total 31 numeric, 6 one-hot and 5 target-encoded columns plus 256 embedding dimensions (356 features "
      "after encoding). Counts are log-transformed, numeric features standardized, and missing enrichment values "
      "imputed with a missing-indicator. All encoders are fitted on training folds only.", small),

    P("4. Baseline models (Stanford's setup, rebuilt on our split)", h),
    P("Inputs: title embedding + year + title length + journal integer code, as in the Stanford report. "
      "<b>Multinomial Naive Bayes</b> (min-max scaled inputs), <b>random forest</b> (200 fully grown trees) and a "
      "<b>neural network</b> (MLP with 256 and 128 hidden units, early stopping), plus the <b>year-only rule</b>. "
      "Our rebuild reproduces their findings: NN test F1 0.7095 (theirs 0.7026) and a heavily overfit forest "
      "(train 1.000, test 0.655)."),

    P("5. Models we built and trained", h),
    Paragraph('<b>SVM – six kernel / transformation variants</b> on scaled features: linear; exact RBF (cost grows with n<super>2</super>, so trained on a 30k subsample); Nyström RBF approximation and random Fourier features (both use all rows); Nyström polynomial (degree 2); polynomial count sketch. Standard, quantile and power scaling and class weighting were also compared.', bullet, bulletText="•"),
    Paragraph('<b>LightGBM</b> – gradient-boosted decision trees built one after another, each correcting the previous ones; regularized with small trees (63 leaves), row/column subsampling and early stopping.', bullet, bulletText="•"),
    Paragraph('<b>Ordinal LightGBM</b> – the bins are ordered, so a LightGBM <i>regressor</i> predicts log(1 + citations) and two thresholds, tuned on out-of-fold CV predictions, cut the score into the three bins; neighbouring-bin mistakes therefore cost less than distant ones.', bullet, bulletText="•"),
    Paragraph('<b>MLP</b> – neural network with two hidden layers (512, 256), L2 penalty and early stopping.', bullet, bulletText="•"),
    Paragraph('<b>Stacking</b> – a logistic regression trained on the out-of-fold outputs of the four models above.', bullet, bulletText="•"),
    P("All models use the same 5 CV folds; the final model was chosen by CV score."),
]

story += [
    KeepTogether([
    P("6. Results (test set, used once)", h),
    table([
        ["Model", "Features", "Train F1", "CV F1", "Test macro F1", "Test weighted F1"],
        ["Ordinal LightGBM (ours, best)", "full", "0.855", "0.727", "0.7313", "0.7233"],
        ["Stacking ensemble (ours)", "full", "0.806", "0.727", "0.7299", "0.7226"],
        ["Ordinal LightGBM (ours)", "base (no enrichment)", "0.845", "0.724", "0.7268", "0.7188"],
        ["LightGBM classifier (ours)", "full", "0.920", "0.711", "0.7161", "0.7080"],
        ["SVM, linear – best kernel (ours)", "full", "0.742", "0.708", "0.7116", "0.7039"],
        ["MLP (ours)", "full", "0.761", "0.701", "0.7024", "0.6948"],
        ["Stanford NN – reported in their paper", "Stanford", "0.7027", "–", "0.7026", "–"],
        ["Stanford NN – rebuilt by us", "Stanford", "0.713", "–", "0.7095", "0.7020"],
        ["Year-only rule", "year", "0.697", "–", "0.6998", "0.6923"],
        ["Stanford RF – rebuilt by us", "Stanford", "1.000", "–", "0.6546", "0.6513"],
        ["Stanford NB – rebuilt by us", "Stanford", "0.354", "–", "0.3557", "0.3749"],
    ], [5.4, 3.0, 1.6, 1.4, 2.3, 2.6], bold_rows=(1,)),
    P("F1 = macro F1 unless stated (Stanford did not say which average they used). SVM CV macro F1 by variant "
      "(full features): linear 0.708, Nyström polynomial 0.707, Nyström RBF 0.706, exact RBF 0.693 "
      "(train 0.885 – overfits), random Fourier 0.657, polynomial sketch 0.412; scaler choice made no difference.",
      small)]),

    KeepTogether([
        P("7. Confusion matrices (test set; rows = actual bin, share of each actual bin, counts in brackets)", h),
        Image(str(ROOT / "reports" / "confusion_matrices.png"), width=17.6 * cm, height=17.6 * cm * 660 / 2100)]),
    P("All models separate the two extreme bins well; almost no &lt;60 paper is predicted &gt;120. The middle bin is "
      "the hardest for every model (about 66% correct), as Stanford also found. Our model's gain comes mainly from "
      "the <b>&gt;120 bin</b>: correctly identified highly cited papers rise from 60.3% (Stanford-style NN) to "
      "<b>67.7%</b>, and &gt;120 papers mistaken for &lt;60 fall from 3.1% to 1.3%."),

    P("8. Implementation", h),
    P("Python 3.12 with pandas, scikit-learn, LightGBM, PyTorch (Apple-Silicon GPU) and sentence-transformers. "
      "The pipeline is a sequence of scripts (download → clean → Crossref/OpenAlex enrichment → split → "
      "embeddings → baselines → features → SVM comparison → final models → report); every score is logged to "
      "<font face='Courier'>reports/results.csv</font>. The final models are saved to "
      "<font face='Courier'>models/full_orig.joblib</font>, and <font face='Courier'>src/predict.py</font> loads them "
      "for the live demo in seconds. Results are reproducible (fixed split and seeds; LightGBM fixed to 6 threads)."),

    P("9. Conclusions", h),
    Paragraph("Our best model, <b>ordinal LightGBM on the enriched features, reaches test macro F1 0.7313</b> (weighted 0.7233) – <b>+0.029 over Stanford's 0.7026</b> and +0.022 over their method rebuilt on the same split.", bullet, bulletText="•"),
    Paragraph('<b>No problematic overfitting:</b> CV and test scores agree within about 0.005 for every model (best model: CV 0.727, test 0.731), unlike the Stanford random forest (train 1.000, test 0.655).', bullet, bulletText="•"),
    Paragraph('<b>What helped most:</b> treating the bins as ordered (+0.015 over a plain classifier on the same features) and target-encoding journals. The OpenAlex features added a smaller +0.005: in a dataset of only highly cited papers, author, reference and funding counts barely differ between papers.', bullet, bulletText="•"),
    Paragraph('<b>SVM:</b> with good embeddings the data is close to linearly separable; kernels did not beat a linear SVM, and the exact RBF kernel overfit (train 0.885, CV 0.693).', bullet, bulletText="•"),
    Paragraph('<b>Limitation and future work:</b> year alone explains most of this task (0.700), so the headroom is small. Next steps would be author track record measured at publication time and full-text features.', bullet, bulletText="•"),
]


def footer(canvas, doc):
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawRightString(A4[0] - 1.4 * cm, 0.8 * cm, f"Page {doc.page} of 2")


doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.4 * cm, rightMargin=1.4 * cm,
                        topMargin=1.1 * cm, bottomMargin=1.2 * cm, title="Paper Citation Classification – Write-up",
                        author=TEAM)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(f"Saved {OUT}")
