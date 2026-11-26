"""
analyze_results.py: report-ready tables from the screener's results (V4 categories).

  !python analyze_results.py --results results_v4

Creates in the results folder:
  summary_by_patient.csv   counts per patient and category (report table)
  all_assessments.csv      one row per patient-trial pair
"""

import argparse
import csv
import glob
import json
import os
from collections import Counter

CATS = ["Eligible", "Likely eligible - needs review", "Likely ineligible - needs review",
        "Not eligible", "Unable to assess - screening error"]
SHORT = dict(zip(CATS, ["Eligible", "Likely elig.", "Likely inel.", "Not eligible", "Error"]))


def write(path, rows):
    if rows:
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_v4")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.results, "P*.json")))
    if not files:
        raise SystemExit(f"No patient results in {args.results}")

    summary, rows = [], []
    print(f"{'Patient':<8}{'Screened':>9}" + "".join(f"{SHORT[c]:>14}" for c in CATS) + "  Complete")
    for f in files:
        d = json.load(open(f))
        cats = Counter(r["category"] for r in d["results"])
        pid = d["patient_id"]
        summary.append({"patient_id": pid, "label": d.get("label"),
                        "trials_in_database": d["funnel"]["all_trials"],
                        "after_rules": d["funnel"]["after_prefilter"],
                        "screened": len(d["results"]),
                        **{SHORT[c]: cats[c] for c in CATS},
                        "complete": d.get("complete", True),
                        "models": ", ".join(d.get("experiment", {}).get("models_actually_used", []))})
        print(f"{pid:<8}{len(d['results']):>9}" + "".join(f"{cats[c]:>14}" for c in CATS)
              + f"  {d.get('complete', True)}")
        for rank, r in enumerate(d["results"], 1):
            rows.append({"patient_id": pid, "rank": rank, "nct_id": r["nct_id"],
                         "title": r["title"], "result": r["category"],
                         "pass": r["n_pass"], "fail": r["n_fail"], "unknown": r["n_unknown"],
                         "unknown_important": r.get("n_unknown_important"),
                         "fit_pct": r["fit_score"], "model": r.get("model"), "url": r["url"]})

    tot = Counter(r["result"] for r in rows)
    print(f"\nTotal: {len(rows)} assessments | " + " | ".join(f"{SHORT[c]}: {tot[c]}" for c in CATS))
    write(os.path.join(args.results, "summary_by_patient.csv"), summary)
    write(os.path.join(args.results, "all_assessments.csv"), rows)
    print(f"Saved summary_by_patient.csv and all_assessments.csv in {args.results}/")


if __name__ == "__main__":
    main()
