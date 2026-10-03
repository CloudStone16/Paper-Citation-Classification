"""Summarize reports/results.csv into reports/results.md and the README results section."""
import pandas as pd

from config import ROOT

STANFORD = [("Stanford NB (reported)", 0.4796, 0.4783), ("Stanford RF (reported)", 0.9999, 0.6336),
            ("Stanford NN (reported)", 0.7027, 0.7026)]


def table(res, task):
    r = res[res.task == task].copy()
    # Group CV / train / test rows of the same run: key = model + embeddings (first word of notes)
    r["notes"] = r["notes"].fillna("").str.split().str[0].fillna("")
    r.loc[r.model.str.startswith("Stanford") | r.model.str.startswith("year-only"), "notes"] = "baseline"
    r = r.sort_values("time").drop_duplicates(["model", "split", "notes"], keep="last")
    wide = r.pivot_table(index=["model", "notes"], columns="split", values="macro_f1", aggfunc="last")
    wf1 = r[r.split == "test"].drop_duplicates(["model", "notes"], keep="last").set_index(["model", "notes"]).weighted_f1
    wide["test_weighted_f1"] = wf1
    wide = wide.reindex(columns=[c for c in ("train", "cv", "test", "test_weighted_f1") if c in wide])
    if "train" in wide and "test" in wide:
        wide["train-test gap"] = wide["train"] - wide["test"]
    wide = wide.reset_index()
    wide = wide.rename(columns={"notes": "inputs"})
    sort_col = "test" if "test" in wide else "cv"
    return wide.sort_values(sort_col, ascending=False, na_position="last").round(4).fillna("–")


def svm_table():
    """Best CV macro F1 per SVM variant (standard scaler) for each feature set and task."""
    path = ROOT / "reports" / "svm_cv.csv"
    if not path.exists():
        return ""
    # Rows have different numbers of hyperparameter columns per variant (e.g. linear has no gamma),
    # but always end with: scaler, train_f1, cv_macro_f1, cv_weighted_f1, cv_accuracy, minutes
    import csv
    rows = []
    with open(path) as f:
        for rec in csv.reader(f):
            if rec[0] == "setup":
                continue
            setup, variant, *params = rec[:-6]
            scaler, train_f1, cv_f1 = rec[-6], float(rec[-5]), float(rec[-4])
            rows.append({"setup": setup, "variant": variant, "params": " ".join(params), "scaler": scaler,
                         "train_f1": train_f1, "cv_macro_f1": cv_f1})
    d = pd.DataFrame(rows)
    d = d[(d.scaler == "standard") & ~d.params.str.contains("balanced") & d.setup.str.endswith("|orig")]
    d["setup"] = d.setup.str.split("|").map(lambda x: f"{x[0]} features")
    best = d.sort_values("cv_macro_f1").groupby(["setup", "variant"]).tail(1)
    cv = best.pivot(index="variant", columns="setup", values="cv_macro_f1")
    tr = best.pivot(index="variant", columns="setup", values="train_f1")
    out = cv.round(4).astype(str) + " (train " + tr.round(3).astype(str) + ")"
    order = ["linear", "rbf_exact", "rbf_nystroem", "rbf_fourier", "poly_nystroem", "poly_sketch"]
    return out.reindex([v for v in order if v in out.index]).to_markdown()


def main():
    res = pd.read_csv(ROOT / "reports" / "results.csv")
    parts = ["## Results", "",
             "See [`docs/RESULTS_SUMMARY.md`](docs/RESULTS_SUMMARY.md) for a plain-language summary and "
             "[`docs/METHODS.md`](docs/METHODS.md) for how everything works.", "",
             "Macro F1 unless noted. `cv` = mean over the 5 training folds; `test` = held-out 15%, used once.",
             "", "### Original task (bins <60 / 60-120 / >120)", "",
             "| Reported by Stanford | Train F1 | Test F1 |", "|---|---|---|"]
    parts += [f"| {n} | {a} | {b} |" for n, a, b in STANFORD]
    parts += ["", table(res, "orig").to_markdown(index=False), "",
              "### SVM kernel / transformation comparison (best CV macro F1 per variant)", "",
              "`rbf_exact` was trained on a 30k subsample (its cost grows with n^2); all others on all training rows.",
              "", svm_table(), ""]
    md = "\n".join(parts)
    (ROOT / "reports" / "results.md").write_text(md)

    readme = (ROOT / "README.md").read_text()
    head = readme.split("## Results")[0]
    (ROOT / "README.md").write_text(head + md)
    print(md)


if __name__ == "__main__":
    main()
