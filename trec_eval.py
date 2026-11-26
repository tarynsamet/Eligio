"""
trec_eval.py: measures the screener against PHYSICIAN labels (TREC 2021 Clinical Trials track).

TREC is a public research benchmark run by NIST. Physicians judged real patient-trial
pairs as 2 = eligible, 1 = excluded, 0 = not relevant. We screen a sample of those pairs
with our own screener prompt and report how often our result agrees with the physicians.

  !python -u trec_eval.py --pairs 50

Uses about one Gemini request per pair. Saves after every pair and resumes if re-run.
Outputs (folder trec_eval/): trec_pairs.csv (every pair) and trec_summary.txt (the numbers).

Caveats to state in the report:
- TREC topics are synthetic admission notes covering many diseases, not just breast cancer
  (we prefer cancer-related topics where possible).
- Labels were made against an April 2021 snapshot of each trial; we fetch the trial's
  CURRENT criteria from ClinicalTrials.gov, which may have been amended since.
"""

import argparse
import csv
import json
import os
import random
import re
import time

import requests

import screener
from fetch_trials import clean_study

TOPICS_URL = "https://trec.nist.gov/data/trials/topics2021.xml"
QRELS_URL = "https://trec.nist.gov/data/trials/qrels2021.txt"
API = "https://clinicaltrials.gov/api/v2/studies/"
OUT = "trec_eval_v5"
LABEL = {2: "eligible", 1: "excluded", 0: "not relevant"}
CANCER = re.compile(r"cancer|carcinoma|tumou?r|lymphoma|leukemia|melanoma|metasta|sarcoma|"
                    r"myeloma|neoplasm|malignan|glioma", re.I)


def get_text(url, local):
    if os.path.exists(local):
        return open(local, encoding="utf-8").read()
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    open(local, "w", encoding="utf-8").write(r.text)
    return r.text


def load_topics(xml):
    return {int(n): re.sub(r"\s+", " ", t).strip()
            for n, t in re.findall(r'<topic number="(\d+)"[^>]*>(.*?)</topic>', xml, re.S)}


def load_qrels(txt):
    out = []
    for line in txt.splitlines():
        p = line.split()
        if len(p) >= 4 and p[2].startswith("NCT"):
            out.append((int(p[0]), p[2], int(p[3])))
    return out


def sample_pairs(topics, qrels, n, seed=42, exclude_topics=()):
    """Balanced sample: 40% eligible, 40% excluded, 20% not relevant; cancer topics first;
    at most 3 pairs per topic so many different patients are covered.
    exclude_topics: patients used before (for a held-out test on new patients)."""
    rng = random.Random(seed)
    topics = {t: v for t, v in topics.items() if t not in set(exclude_topics)}
    want = {2: round(n * 0.4), 1: round(n * 0.4)}
    want[0] = n - want[2] - want[1]
    cancer_topics = {t for t, txt in topics.items() if CANCER.search(txt)}
    picked, per_topic = [], {}
    for label in (2, 1, 0):
        pool = [q for q in qrels if q[2] == label and q[0] in topics]
        rng.shuffle(pool)
        pool.sort(key=lambda q: q[0] not in cancer_topics)   # cancer topics first
        for t, nct, lab in pool:
            if sum(1 for p in picked if p[2] == label) >= want[label]:
                break
            if per_topic.get(t, 0) >= 3:
                continue
            picked.append((t, nct, lab))
            per_topic[t] = per_topic.get(t, 0) + 1
    return picked


def fetch_trial(nct):
    r = requests.get(API + nct, params={"format": "json"}, timeout=30)
    if r.status_code != 200:
        return None
    t = clean_study(r.json())
    return t if t.get("criteria_raw") else None


def predicted(category):
    if category in (screener.ELIGIBLE, screener.LIKELY_ELIGIBLE):
        return "eligible"
    if category in (screener.LIKELY_INELIGIBLE, screener.NOT_ELIGIBLE):
        return "not eligible"
    return "error"


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=50)
    ap.add_argument("--out", default=OUT, help="results folder (new folder = fresh run)")
    ap.add_argument("--compare", default="trec_results_v4.json",
                    help="earlier trec_results.json to compare against, if present")
    ap.add_argument("--seed", type=int, default=42, help="sampling seed (42 = original sample)")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="earlier trec_results*.json files: their patients (topics) are left out, "
                         "so the test uses only NEW patients and trials")
    args = ap.parse_args()
    OUT = args.out
    os.makedirs(OUT, exist_ok=True)

    topics = load_topics(get_text(TOPICS_URL, f"{OUT}/topics2021.xml"))
    qrels = load_qrels(get_text(QRELS_URL, f"{OUT}/qrels2021.txt"))
    print(f"TREC 2021: {len(topics)} patient topics, {len(qrels)} physician-labeled pairs")
    if not topics or not qrels:
        raise SystemExit("Could not read the TREC files (format changed?).")

    done_path = f"{OUT}/trec_results.json"
    done = json.load(open(done_path)) if os.path.exists(done_path) else {}
    used = set()
    for path in args.exclude:
        if os.path.exists(path):
            used |= {int(k.split("|")[0]) for k in json.load(open(path))}
        else:
            raise SystemExit(f"--exclude file not found: {path}")
    if used:
        print(f"Held-out test: leaving out {len(used)} patients used in earlier runs")
    pairs = sample_pairs(topics, qrels, args.pairs, seed=args.seed, exclude_topics=used)
    print(f"Sample: {len(pairs)} pairs ({sum(p[2]==2 for p in pairs)} eligible, "
          f"{sum(p[2]==1 for p in pairs)} excluded, {sum(p[2]==0 for p in pairs)} not relevant); "
          f"{sum(f'{t}|{n}' in done for t, n, _ in pairs)} already done")
    print(f"Model: {screener.MODEL}\n")

    for i, (t, nct, lab) in enumerate(pairs, 1):
        key = f"{t}|{nct}"
        if key in done and done[key]["predicted"] != "error":
            continue
        trial = fetch_trial(nct)
        if not trial:
            print(f"({i}/{len(pairs)}) topic {t} / {nct}: trial not retrievable, skipped")
            done[key] = {"topic": t, "nct_id": nct, "label": LABEL[lab], "category": "skipped",
                         "predicted": "skipped"}
            continue
        print(f"({i}/{len(pairs)}) topic {t} / {nct} (physicians: {LABEL[lab]})...", flush=True)
        try:
            r = screener.screen_one(topics[t], trial)     # V5: retries malformed answers
        except screener.DailyQuotaExceeded:
            json.dump(done, open(done_path, "w"), indent=1)
            raise SystemExit("Daily request limit reached; progress saved. Re-run later to resume.")
        except Exception as e:
            r = {"criteria": [], "summary": f"failed: {e}"}
        sc = screener.score_checklist(r)
        fails = [c["criterion"] for c in r.get("criteria", []) if c.get("status") == "FAIL"]
        done[key] = {"topic": t, "nct_id": nct, "label": LABEL[lab], "category": sc["category"],
                     "predicted": predicted(sc["category"]), "trial_title": trial["title"],
                     "fail_reasons": " | ".join(fails)[:500], "summary": r.get("summary", "")[:500],
                     "error": (r.get("summary", "") if sc["category"] == screener.ERROR_CATEGORY else "")[:300]}
        print(f"      -> {sc['category']}")
        json.dump(done, open(done_path, "w"), indent=1)
        time.sleep(0.5)

    # ------------------------------------------------------------- scoring
    rows = [done[f"{t}|{n}"] for t, n, _ in pairs if f"{t}|{n}" in done]
    rows = [r for r in rows if r["predicted"] not in ("skipped", "error")]
    with open(f"{OUT}/trec_pairs.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["topic", "nct_id", "label", "category", "predicted",
                                          "trial_title", "fail_reasons", "summary", "error"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    def truth(r):
        return "eligible" if r["label"] == "eligible" else "not eligible"

    n = len(rows)
    agree = sum(truth(r) == r["predicted"] for r in rows)
    tp = sum(truth(r) == "eligible" and r["predicted"] == "eligible" for r in rows)
    fn = sum(truth(r) == "eligible" and r["predicted"] != "eligible" for r in rows)
    tn = sum(truth(r) != "eligible" and r["predicted"] != "eligible" for r in rows)
    fp = sum(truth(r) != "eligible" and r["predicted"] == "eligible" for r in rows)
    pct = lambda a, b: f"{100 * a / b:.0f}%" if b else "n/a"

    cats = [screener.ELIGIBLE, screener.LIKELY_ELIGIBLE, screener.LIKELY_INELIGIBLE,
            screener.NOT_ELIGIBLE]
    lines = [f"TREC 2021 physician labels vs our screener ({screener.MODEL}), {n} pairs scored"
             + (f" [HELD-OUT: new patients only, seed {args.seed}]" if used else ""),
             "",
             f"Overall agreement (eligible vs not eligible): {agree}/{n} = {pct(agree, n)}",
             f"Eligible pairs found (sensitivity):           {tp}/{tp + fn} = {pct(tp, tp + fn)}",
             f"Ineligible pairs ruled out (specificity):     {tn}/{tn + fp} = {pct(tn, tn + fp)}",
             f"When we say eligible, physicians agree (PPV): {tp}/{tp + fp} = {pct(tp, tp + fp)}",
             "",
             "Our category (rows) vs physician label (columns):",
             f"{'':<36}{'eligible':>10}{'excluded':>10}{'not relev.':>11}"]
    for c in cats:
        lines.append(f"{c:<36}" + "".join(
            f"{sum(r['category'] == c and r['label'] == l for r in rows):>{w}}"
            for l, w in (("eligible", 10), ("excluded", 10), ("not relevant", 11))))
    # V5 additions: technical failure rate, and the "hidden from the coordinator" view
    attempted = [done[f"{t}|{n_}"] for t, n_, _ in pairs if f"{t}|{n_}" in done
                 and done[f"{t}|{n_}"]["predicted"] != "skipped"]
    n_err = sum(r["predicted"] == "error" for r in attempted)
    elig = [r for r in rows if r["label"] == "eligible"]
    hidden = sum(r["category"] == screener.NOT_ELIGIBLE for r in elig)
    lines += ["",
              f"Technical failures (no usable answer): {n_err}/{len(attempted)} attempted pairs "
              f"= {pct(n_err, len(attempted))} (excluded from the numbers above)",
              f"Physician-eligible pairs marked 'Not eligible' (i.e. not flagged for review): "
              f"{hidden}/{len(elig)} = {pct(hidden, len(elig))}",
              f"Physician-eligible pairs still shown for review (any category except Not eligible): "
              f"{len(elig) - hidden}/{len(elig)} = {pct(len(elig) - hidden, len(elig))}"]
    if os.path.exists(args.compare) and not used:      # no overlap on a held-out test
        old = json.load(open(args.compare))
        changed = [(k, old[k]["category"], done[k]["category"], done[k]["label"])
                   for k in done if k in old and old[k].get("category") != done[k].get("category")]
        lines += ["", f"Changes vs {args.compare}: {len(changed)} pair(s) changed category"]
        lines += [f"  {k:<16} {a} -> {b}   (physicians: {l})" for k, a, b, l in changed]
    text = "\n".join(lines)
    open(f"{OUT}/trec_summary.txt", "w").write(text + "\n")
    print("\n" + text + f"\n\nSaved {OUT}/trec_pairs.csv and {OUT}/trec_summary.txt")


if __name__ == "__main__":
    main()
