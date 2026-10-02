"""Summarize reports/results.csv into reports/results.md and the README results section."""
import pandas as pd

from config import ROOT

STANFORD = [("Stanford NB (reported)", 0.4796, 0.4783), ("Stanford RF (reported)", 0.9999, 0.6336),
            ("Stanford NN (reported)", 0.7027, 0.7026)]


def table(res, task):
    r = res[res.task == task]
    # Latest entry per (model, split, notes) wins
    r = r.sort_values("time").drop_duplicates(["model", "split", "notes"], keep="last")
    wide = r.pivot_table(index=["model", "notes"], columns="split", values="macro_f1", aggfunc="last")
    wf1 = r[r.split == "test"].set_index(["model", "notes"]).weighted_f1
    wide["test_weighted_f1"] = wf1
    wide = wide.reindex(columns=[c for c in ("train", "cv", "test", "test_weighted_f1") if c in wide])
    if "train" in wide and "test" in wide:
        wide["train-test gap"] = wide["train"] - wide["test"]
    wide = wide.reset_index()
    wide["notes"] = wide["notes"].fillna("").str.slice(0, 60)
    sort_col = "test" if "test" in wide else "cv"
    return wide.sort_values(sort_col, ascending=False, na_position="last").round(4)


def main():
    res = pd.read_csv(ROOT / "reports" / "results.csv")
    parts = ["## Results", "",
             "Macro F1 unless noted. `cv` = mean over the 5 training folds; `test` = held-out 15%, used once.",
             "", "### Original task (bins <60 / 60-120 / >120)", "",
             "| Reported by Stanford | Train F1 | Test F1 |", "|---|---|---|"]
    parts += [f"| {n} | {a} | {b} |" for n, a, b in STANFORD]
    parts += ["", table(res, "orig").to_markdown(index=False), "",
              "### Within-year task (bottom / middle / top third of each year; chance = 0.333)", "",
              table(res, "yn").to_markdown(index=False), ""]
    md = "\n".join(parts)
    (ROOT / "reports" / "results.md").write_text(md)

    readme = (ROOT / "README.md").read_text()
    head = readme.split("## Results")[0]
    (ROOT / "README.md").write_text(head + md)
    print(md)


if __name__ == "__main__":
    main()
