"""
screener.py — AI eligibility screener for the breast cancer trial matcher.

Pipeline (one patient):
  1. PARSE      LLM turns the free-text patient summary into structured facts
                (age, sex, subtype, disease setting, biomarkers, ...).
  2. PRE-FILTER Cheap rules on trials.json: treatment trials only, sex, age.
  3. RANK       Keyword relevance score (subtype, setting, biomarkers) and
                keep the top N trials for the expensive step.
  4. SCREEN     LLM reads each trial's inclusion/exclusion text and returns a
                criterion-by-criterion checklist (PASS / FAIL / UNKNOWN).
  5. SCORE      Ranking is computed from the checklist (not the LLM's opinion):
                any FAIL -> likely ineligible; otherwise more PASS = higher.
  6. REPORT     Saves results/<patient>.json and a readable results/<patient>.md

VERSION 4 (Nov 22): four coordinator-facing categories, trial-purpose check,
stricter 'Not eligible', routine vs important unknowns. Builds on V3 (Grace's V2
safeguards + resume, daily-quota stop, site status).

Run in Colab:
  !pip install -q google-genai
  !python screener.py --patient P01          # one patient
  !python screener.py                        # all patients in patients.json

API key: add GEMINI_API_KEY in Colab's Secrets panel (key icon, left sidebar)
and switch "Notebook access" on. Never paste the key into this file.
"""

import argparse
import json
import os
import re
import time

# ---------------------------------------------------------------- settings
TRIALS_FILE = "trials.json"
PATIENTS_FILE = "patients.json"
RESULTS_DIR = "results"

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")  # override: set GEMINI_MODEL
# Optional backup models used only if the main one keeps returning 503 (overloaded),
# e.g. GEMINI_FALLBACK="gemini-3.1-flash-lite". Each result records the model actually used.
FALLBACK_MODELS = [m.strip() for m in os.environ.get("GEMINI_FALLBACK", "").split(",") if m.strip()]
REQUEST_TIMEOUT_S = 90         # give up on a request that hangs
TOP_N = 15                     # trials sent to the LLM per patient
SECONDS_BETWEEN_CALLS = 5      # free tier has a per-minute request limit
MAX_RETRIES = 4


# ================================================================ prompts
# These are the two prompts the system uses (include them in report item 3).

PARSER_PROMPT = """You are a clinical data abstractor for breast cancer clinical trial screening.
Extract structured facts from the patient summary. Use only what is stated.
If something is not stated, use null (or an empty list). Never guess.

Return JSON with exactly these keys:
{
  "age": number or null,
  "sex": "FEMALE" | "MALE" | null,
  "menopausal_status": "premenopausal" | "postmenopausal" | null,
  "subtype": "HR+/HER2-" | "HER2+" | "TNBC" | "unknown",
  "her2_low": true | false | null,
  "disease_setting": "early" | "metastatic" | "unknown",
  "biomarkers": [list of molecular alterations, e.g. "PIK3CA H1047R", "germline BRCA1", "ESR1 Y537S", "PD-L1 CPS 15"],
  "prior_therapies": [list of prior cancer treatments, drug names where given],
  "lines_of_therapy_metastatic": number or null,
  "ecog": number or null,
  "brain_metastases": "none" | "treated_stable" | "active" | "unknown",
  "notable_comorbidities": [list],
  "missing_information": [important screening facts NOT in the summary]
}

Definitions: HER2+ means IHC 3+ or ISH amplified (HR status may be positive or negative).
HR+/HER2- means ER and/or PR positive and HER2 not amplified (includes HER2-low).
TNBC means ER, PR and HER2 all negative."""


SCREENER_PROMPT = """You are an assistant to a clinical research coordinator at a New York City cancer center.
Your job: check one patient against one clinical trial's eligibility criteria.

Rules:
1. Work only from the patient summary and the trial criteria provided. Never assume facts that are not stated.
2. Evaluate the clinically important criteria: diagnosis and subtype, disease stage/setting,
   biomarkers, prior therapies and lines of therapy, performance status, brain metastases,
   organ function/labs, and key comorbidities. Skip boilerplate (consent, contraception,
   willingness to comply) unless it is decisive. List at most 15 criteria.
   Always prioritize mandatory criteria that could independently disqualify the patient.
3. For EACH criterion give a status from the patient's point of view:
     PASS    = the patient satisfies it (for an exclusion criterion: the patient does NOT have the excluded condition)
     FAIL    = the patient clearly does not satisfy it (for an exclusion criterion: the patient HAS the excluded condition)
     UNKNOWN = the summary does not contain enough information to decide
   When in doubt, use UNKNOWN, never PASS.
4. Quote or closely paraphrase the patient evidence for each decision. For UNKNOWN, say what is missing.
5. "open_questions" are specific things the care team must check to confirm eligibility.
6. If the trial is for a cohort/arm, judge against the cohort that best fits the patient and name it in the summary.

Additional clinical safeguards:
7. Distinguish neoadjuvant (before surgery), adjuvant (after surgery), and metastatic
   treatment settings. Evaluate whether prior surgery or treatment timing conflicts
   with the trial's requirements.
8. Evaluate mandatory biomarkers and any treatment-history requirements tied to
   those biomarkers. Check BOTH inclusion and exclusion criteria.
9. Never mark a criterion PASS based only on missing information.
   Never mark a criterion FAIL simply because a test or treatment is not mentioned.
   Use UNKNOWN when evidence is insufficient.
10. For numerical or time-sensitive criteria (such as LVEF, laboratory values, and
    prior treatment intervals), check both the value and the relevant date.
11. For every PASS and FAIL, verify that the supporting evidence is consistent
    with the chosen status. If uncertain, use UNKNOWN.
12. For trials with multiple cohorts, assess the most applicable cohort without
    combining requirements from different cohorts.
13. Prioritize decisive eligibility requirements when selecting the 15 criteria.

Check what the trial is FOR (Version 4):
14. First identify the trial's core target: the condition, disease setting and
    population it is designed for (e.g. leptomeningeal metastasis; neoadjuvant
    treatment before surgery; PD-L1-negative disease; a specific mutation).
    Always include that core requirement as one of your criteria.
15. If the summary describes the patient's disease, disease sites and treatment
    history, and the patient clearly does NOT have the trial's core target
    condition or setting, mark that criterion FAIL. This is the one exception to
    rule 9: a detailed disease description is evidence of what the patient has.
    Examples: a trial for leptomeningeal metastasis when only treated brain
    metastases are described; a pre-surgery (neoadjuvant) trial when the patient
    has already had surgery; a trial for patients ineligible for PD-1 therapy when
    the patient's PD-L1 result makes them eligible for it.

For every FAIL, assign a fail_confidence:
- HIGH: A documented patient fact directly contradicts an unambiguous mandatory
  requirement for the applicable trial cohort.
- LOW: A possible mismatch depends on uncertain timing, interpretation, missing
  information, or ambiguous cohort applicability.
- A test that was not done or not reported is NEVER a HIGH fail: use UNKNOWN
  (important) instead.
- For PASS or UNKNOWN, use null.

For every UNKNOWN, assign an importance:
- important: the missing fact could decide eligibility (an untested biomarker,
  unclear treatment history or timing, disease measurability, a cohort question).
- routine: standard screening items that sites check for nearly every patient
  (routine labs and organ function, pregnancy test, consent, contraception,
  vaccination, recent infections, ECG/QTc, weight).
- For PASS or FAIL, use null.

A HIGH fail_confidence is an AI assessment, not independently confirmed clinical
ineligibility. Do not invent evidence to justify HIGH confidence.

Return JSON with exactly these keys:
{
  "summary": "2-3 sentences: overall fit and deciding factors",
  "criteria": [
    {
      "criterion": "short restatement",
      "type": "inclusion" | "exclusion",
      "status": "PASS" | "FAIL" | "UNKNOWN",
      "evidence": "patient evidence or what is missing",
      "fail_confidence": "HIGH" | "LOW" | null,
      "importance": "important" | "routine" | null
    }
  ],
  "open_questions": ["..."]
}"""



# ================================================================ LLM call
# The only provider-specific code. To switch to Claude or OpenAI, change this.
_client = None


def _get_api_key():
    try:  # Colab Secrets panel
        from google.colab import userdata
        key = userdata.get("GEMINI_API_KEY")
        if key:
            return key
    except Exception:
        pass
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("No GEMINI_API_KEY found. Add it as an environment variable "
                           "(Render) or in Colab Secrets (key icon, notebook access on).")
    return key


class DailyQuotaExceeded(Exception):
    """The free-tier DAILY request limit is used up: retrying today will not help."""


def _is_daily_quota(msg):
    return ("429" in msg or "RESOURCE_EXHAUSTED" in msg) and \
        ("PerDay" in msg or "per day" in msg or "requests per day" in msg.lower())


LAST_MODEL_USED = None


def call_llm(system_prompt, user_text):
    """Send one request; return the parsed JSON dict.
    - daily quota used up  -> raise DailyQuotaExceeded at once (no pointless retries)
    - per-minute limit / 503 overload / timeout -> wait and retry
    - repeated 503 -> try FALLBACK_MODELS, if any were set
    """
    global _client, LAST_MODEL_USED
    from google import genai
    from google.genai import types

    if _client is None:
        _client = genai.Client(
            api_key=_get_api_key(),
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_S * 1000))

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        temperature=0,
    )
    models = [MODEL] + FALLBACK_MODELS
    for model in models:
        for attempt in range(MAX_RETRIES):
            try:
                resp = _client.models.generate_content(
                    model=model, contents=user_text, config=config)
                time.sleep(SECONDS_BETWEEN_CALLS)
                LAST_MODEL_USED = model
                return parse_json(resp.text)
            except json.JSONDecodeError:
                raise
            except Exception as e:
                msg = str(e)
                if _is_daily_quota(msg):
                    raise DailyQuotaExceeded(f"{model}: {msg[:300]}")
                overloaded = "503" in msg or "UNAVAILABLE" in msg
                retryable = overloaded or any(k in msg for k in (
                    "429", "RESOURCE_EXHAUSTED", "timed out", "Timeout", "ReadTimeout"))
                if not retryable:
                    raise
                if attempt < MAX_RETRIES - 1:
                    wait = 20 * (attempt + 1)
                    print(f"    API problem on {model} (attempt {attempt + 1}): {msg[:200]}")
                    print(f"    waiting {wait}s and retrying...", flush=True)
                    time.sleep(wait)
                    continue
                if overloaded and model != models[-1]:
                    print(f"    {model} still overloaded; switching to backup model", flush=True)
                    break
                raise


def parse_json(text):
    """Parse model output, tolerating ```json fences."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    return json.loads(text)


def validate_screening_response(result):
    """Check Gemini's response before eligibility scoring (Grace, V2)."""
    if not isinstance(result, dict):
        raise ValueError("Screening response must be a dictionary")
    criteria = result.get("criteria")
    if not isinstance(criteria, list) or not criteria:
        raise ValueError("No eligibility criteria returned")
    for i, criterion in enumerate(criteria, start=1):
        if not isinstance(criterion, dict):
            raise ValueError(f"Criterion {i} is not a dictionary")
        if criterion.get("type") not in {"inclusion", "exclusion"}:
            raise ValueError(f"Invalid criterion type at {i}")
        if criterion.get("status") not in {"PASS", "FAIL", "UNKNOWN"}:
            raise ValueError(f"Invalid status at criterion {i}")
        if not isinstance(criterion.get("criterion"), str) or not criterion["criterion"].strip():
            raise ValueError(f"Missing criterion description at {i}")
        if not isinstance(criterion.get("evidence"), str) or not criterion["evidence"].strip():
            raise ValueError(f"Missing evidence at criterion {i}")
        if criterion.get("fail_confidence") not in {"HIGH", "LOW", None}:
            raise ValueError(f"Invalid fail confidence at {i}")
        if criterion.get("importance") not in {"important", "routine", None}:
            raise ValueError(f"Invalid importance at {i}")
    if not isinstance(result.get("open_questions"), list):
        raise ValueError("Missing or invalid open_questions")
    return result


def screen_one(summary, trial, attempts=3):
    """Ask the LLM to screen one trial; retry if the answer is malformed or incomplete.
    DailyQuotaExceeded is passed straight up (retrying today will not help)."""
    last = None
    for i in range(attempts):
        try:
            r = validate_screening_response(call_llm(
                SCREENER_PROMPT, f"PATIENT SUMMARY:\n{summary}\n\n---\n\n{trial_text(trial)}"))
            r["model"] = LAST_MODEL_USED
            return r
        except DailyQuotaExceeded:
            raise
        except (ValueError, KeyError, TypeError) as e:   # bad JSON / failed validation
            last = e
            print(f"      malformed answer for {trial.get('nct_id')} (try {i + 1}/{attempts}): "
                  f"{str(e)[:120]}", flush=True)
    raise ValueError(f"No valid answer after {attempts} tries: {last}")


# ================================================================ step 2: pre-filter
def prefilter(trials, patient):
    """Hard rules that need no AI. Returns (kept, removed_counts)."""
    removed = {"not_treatment": 0, "sex": 0, "age": 0}
    kept = []
    for t in trials:
        if t.get("primary_purpose", "TREATMENT") != "TREATMENT":
            removed["not_treatment"] += 1
            continue
        sex = patient.get("sex")
        if sex and t.get("sex") in ("FEMALE", "MALE") and t["sex"] != sex:
            removed["sex"] += 1
            continue
        age = patient.get("age")
        if age is not None:
            if (t.get("min_age") and age < t["min_age"]) or \
               (t.get("max_age") and age > t["max_age"]):
                removed["age"] += 1
                continue
        kept.append(t)
    return kept, removed


# ================================================================ step 3: rank
RX = {
    "metastatic": r"metastatic|advanced|unresectable|stage iv|recurrent|inoperable",
    "early": r"early[\s-]*(stage|breast)|adjuvant|operable|pathologic complete|"
             r"residual (invasive )?disease|stage (i|ii|iii|ii-iii)\b|dcis|\bsurgery|lumpectomy|mastectomy",
    "hr_pos": r"\b(hr|er|hormone receptor|estrogen receptor)\s*-?\s*(\+|positive)|endocrine",
    "her2_pos": r"her2\s*-?\s*(\+|positive|amplified)|her2-expressing|her2 cancers|her-positive|her2/neu positive",
    "her2_low": r"her2[\s-]*(low|ultralow)",
    "tnbc": r"triple[\s-]*negative|\btnbc\b",
    "solid": r"solid tumou?r|solid malignanc|advanced cancer|advanced malignanc|selected tumor",
}


def _has(pattern, text):
    return re.search(pattern, text, re.I) is not None


def biomarker_terms(biomarkers):
    """'PIK3CA H1047R' -> 'PIK3CA'; 'germline BRCA1' -> 'BRCA1' (gene names only)."""
    genes = set()
    for b in biomarkers or []:
        for tok in re.findall(r"\b[A-Z][A-Z0-9]{2,}\b", b):
            if tok not in {"CPS", "IHC", "ISH", "FISH", "NGS", "DNA", "TPS", "HER2"}:
                genes.add(tok)
    return genes


def relevance_score(trial, p):
    """Keyword score: higher = more likely relevant. Explainable, not clinical."""
    titles = " ".join([trial["title"] or "", trial.get("official_title") or ""])
    head = titles + " " + " ".join(trial.get("conditions", []))
    body = head + " " + (trial.get("inclusion_criteria") or "")[:3000]
    score, reasons = 0, []

    # disease setting: judged on titles (condition lists are noisy);
    # fall back to conditions only if the titles say nothing
    setting = p.get("disease_setting")
    is_met, is_early = _has(RX["metastatic"], titles), _has(RX["early"], titles)
    if is_early and _has(r"neoadjuvant|adjuvant|early", titles):
        is_met = is_met and not _has(r"neoadjuvant|adjuvant", titles)
    if not (is_met or is_early):
        is_met, is_early = _has(RX["metastatic"], head), _has(RX["early"], head)
    if setting == "metastatic":
        if is_met:
            score += 3; reasons.append("metastatic setting")
        elif is_early:
            score -= 4; reasons.append("early-stage trial")
    elif setting == "early":
        if is_early and not is_met:
            score += 3; reasons.append("early-stage setting")
        elif is_met and not is_early:
            score -= 4; reasons.append("metastatic-only trial")

    # subtype (judged on title/conditions)
    sub = p.get("subtype")
    t_hr, t_her2, t_tn = _has(RX["hr_pos"], head), _has(RX["her2_pos"], head), _has(RX["tnbc"], head)
    t_low = _has(RX["her2_low"], head)
    if sub == "HR+/HER2-":
        if t_hr or (t_low and p.get("her2_low")):
            score += 3; reasons.append("HR+ / HER2-low match")
        if (t_tn or t_her2) and not t_hr:
            score -= 4; reasons.append("different subtype")
    elif sub == "TNBC":
        if t_tn:
            score += 3; reasons.append("TNBC match")
        if (t_hr or t_her2) and not t_tn:
            score -= 4; reasons.append("different subtype")
    elif sub == "HER2+":
        if t_her2:
            score += 3; reasons.append("HER2+ match")
        if (t_tn or t_hr) and not t_her2:
            score -= 2; reasons.append("different subtype")

    # biomarkers (anywhere in title or inclusion criteria)
    for gene in biomarker_terms(p.get("biomarkers")):
        if re.search(rf"\b{re.escape(gene)}", body, re.I):
            score += 4; reasons.append(f"{gene} mentioned")

    # breast-specific trials outrank basket trials at the same match level
    if _has(r"breast", titles):
        score += 2; reasons.append("breast-specific")
    elif _has(RX["solid"], titles) and setting == "metastatic":
        score += 1; reasons.append("solid-tumor basket")

    return score, reasons


def rank_trials(trials, patient, top_n=TOP_N):
    scored = []
    for t in trials:
        s, why = relevance_score(t, patient)
        scored.append((s, why, t))
    scored.sort(key=lambda x: (-x[0], -len(x[2].get("nyc_sites", []))))
    return scored[:top_n], scored


# ================================================================ step 5: score
ELIGIBLE = "Eligible"
LIKELY_ELIGIBLE = "Likely eligible - needs review"
LIKELY_INELIGIBLE = "Likely ineligible - needs review"
NOT_ELIGIBLE = "Not eligible"
ERROR_CATEGORY = "Unable to assess - screening error"


def score_checklist(result):
    """Coordinator-facing category from the checklist (V4).
    Not eligible   : a HIGH-confidence FAIL backed by a documented fact
    Likely inel.   : any other FAIL (depends on interpretation or missing info)
    Likely eligible: no FAIL, but an IMPORTANT fact is unknown
    Eligible       : no FAIL, and only routine items are unknown
    """
    crit = result.get("criteria", [])
    empty = {"category": ERROR_CATEGORY, "fit_score": 0, "n_pass": 0, "n_fail": 0,
             "n_unknown": 0, "n_unknown_important": 0}
    if not isinstance(crit, list) or not crit:
        return empty
    if any(not isinstance(c, dict) or c.get("status") not in {"PASS", "FAIL", "UNKNOWN"}
           for c in crit):
        return empty

    n_pass = sum(c["status"] == "PASS" for c in crit)
    n_fail = sum(c["status"] == "FAIL" for c in crit)
    n_unk = sum(c["status"] == "UNKNOWN" for c in crit)
    # an UNKNOWN without a routine tag counts as important (the cautious choice)
    n_unk_imp = sum(c["status"] == "UNKNOWN" and c.get("importance") != "routine" for c in crit)

    if n_fail:
        decisive = any(c["status"] == "FAIL" and c.get("fail_confidence") == "HIGH"
                       and isinstance(c.get("evidence"), str) and c["evidence"].strip()
                       for c in crit)
        category = NOT_ELIGIBLE if decisive else LIKELY_INELIGIBLE
    elif n_unk_imp:
        category = LIKELY_ELIGIBLE
    else:
        category = ELIGIBLE

    fit = round(100 * n_pass / len(crit)) if not n_fail else 0
    return {"category": category, "fit_score": fit, "n_pass": n_pass, "n_fail": n_fail,
            "n_unknown": n_unk, "n_unknown_important": n_unk_imp}


CATEGORY_ORDER = {
    ELIGIBLE: 0,
    LIKELY_ELIGIBLE: 1,
    LIKELY_INELIGIBLE: 2,
    NOT_ELIGIBLE: 3,
    ERROR_CATEGORY: 4,
}



def trial_text(t):
    return (f"TRIAL {t['nct_id']}: {t['title']}\n"
            f"Phase: {'/'.join(t.get('phases', [])) or 'NA'}\n"
            f"Conditions: {', '.join(t.get('conditions', []))}\n"
            f"Interventions: {'; '.join(t.get('interventions', []))}\n\n"
            f"INCLUSION CRITERIA:\n{t.get('inclusion_criteria') or '(none listed)'}\n\n"
            f"EXCLUSION CRITERIA:\n{t.get('exclusion_criteria') or '(none listed)'}")


def _sha(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False)
                          .encode("utf-8")).hexdigest()


def experiment_metadata(trials, patient_record, models_used):
    """Record inputs and settings for reproducibility (Grace, V2)."""
    import hashlib
    from datetime import datetime, timezone
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "configured_model": MODEL,
        "models_actually_used": sorted(m for m in models_used if m),
        "top_n_default": TOP_N,
        "trial_database_sha256": _sha(trials),
        "trial_count": len(trials),
        "patient_record_sha256": _sha(patient_record),
        "parser_prompt_sha256": hashlib.sha256(PARSER_PROMPT.encode("utf-8")).hexdigest(),
        "screener_prompt_sha256": hashlib.sha256(SCREENER_PROMPT.encode("utf-8")).hexdigest(),
    }



def _load_previous(pid, patient_record, trials):
    """Earlier saved results for this patient, if they used the same inputs."""
    path = f"{RESULTS_DIR}/{pid}.json"
    if not os.path.exists(path):
        return None
    try:
        prev = json.load(open(path))
    except Exception:
        return None
    exp = prev.get("experiment", {})
    if exp.get("patient_record_sha256") != _sha(patient_record) or \
            exp.get("trial_database_sha256") != _sha(trials) or \
            exp.get("screener_prompt_sha256") != experiment_metadata([], {}, []).get("screener_prompt_sha256"):
        print("      (saved results used different inputs or prompt; starting fresh)")
        return None
    return prev


def screen_patient(patient_record, trials, top_n=TOP_N, fresh=False):
    pid = patient_record["patient_id"]
    summary = patient_record["summary"]
    print(f"\n=== {pid}: {patient_record.get('label', '')}", flush=True)

    prev = None if fresh else _load_previous(pid, patient_record, trials)
    done = {}
    if prev:
        done = {r["nct_id"]: r for r in prev.get("results", [])
                if r.get("category") != ERROR_CATEGORY}
    models_used = set((prev or {}).get("experiment", {}).get("models_actually_used", []))

    # 1. parse (reused if we already have it)
    if prev and prev.get("profile"):
        profile = prev["profile"]
        print("  [1] reusing saved patient profile")
    else:
        print("  [1] parsing patient summary...", flush=True)
        profile = call_llm(PARSER_PROMPT, f"PATIENT SUMMARY:\n{summary}")
        models_used.add(LAST_MODEL_USED)
    print(f"      subtype={profile.get('subtype')}  setting={profile.get('disease_setting')}  "
          f"biomarkers={profile.get('biomarkers')}")

    # 2. pre-filter
    kept, removed = prefilter(trials, profile)
    print(f"  [2] pre-filter: {len(trials)} -> {len(kept)} trials  (removed: {removed})")

    # 3. rank
    top, _ = rank_trials(kept, profile, top_n)
    print(f"  [3] top {len(top)} by relevance sent to the screener")
    if done:
        print(f"      {sum(t['nct_id'] in done for _, _, t in top)} already screened; resuming")

    def build_out(results, complete):
        results = sorted(results, key=lambda r: (CATEGORY_ORDER.get(r["category"], 9),
                                                 r.get("n_unknown_important", 0), -r["fit_score"]))
        return {"patient_id": pid, "label": patient_record.get("label"),
                "summary": summary, "profile": profile,
                "experiment": experiment_metadata(trials, patient_record, models_used),
                "complete": complete,
                "funnel": {"all_trials": len(trials), "after_prefilter": len(kept),
                           "removed": removed, "screened_by_llm": len(top)},
                "results": results}

    # 4. screen + 5. score  (saved after every trial, so nothing is lost)
    results = []
    for i, (rel, why, t) in enumerate(top, 1):
        if t["nct_id"] in done:
            results.append(done[t["nct_id"]])
            continue
        print(f"  [4] ({i}/{len(top)}) screening {t['nct_id']}...", flush=True)
        try:
            r = screen_one(summary, t)
            models_used.add(r["model"])
        except DailyQuotaExceeded:
            save_outputs(build_out(results, complete=False))
            print(f"  STOPPED: daily request limit reached. {len(results)} of {len(top)} "
                  f"trials saved for {pid}. Re-run the same command tomorrow, or set a "
                  f"different GEMINI_MODEL, and it will resume from here.")
            raise
        except Exception as e:
            print(f"      failed: {str(e)[:200]}")
            r = {"summary": f"Screening failed: {e}", "criteria": [], "open_questions": [],
                 "model": None}
        r.update(score_checklist(r))
        r.update({
            "nct_id": t["nct_id"], "title": t["title"], "url": t["url"],
            "phases": t.get("phases", []), "relevance_score": rel,
            "relevance_reasons": why, "nyc_sites": t.get("nyc_sites", []),
            "central_contact": t.get("central_contact", {}),
        })
        results.append(r)
        save_outputs(build_out(results, complete=False), quiet=True)

    out = build_out(results, complete=True)
    n_err = sum(r["category"] == ERROR_CATEGORY for r in results)
    if n_err:
        print(f"  NOTE: {n_err} trial(s) could not be assessed; re-run to retry only those.")
    save_outputs(out)
    return out


# ================================================================ step 6: report
def borough(zip_code):
    """NYC borough from a zip code (distances are not meaningful within NYC)."""
    z = str(zip_code or "")[:3]
    return {"100": "Manhattan", "101": "Manhattan", "102": "Manhattan", "103": "Staten Island",
            "104": "Bronx", "112": "Brooklyn", "110": "Queens", "111": "Queens",
            "113": "Queens", "114": "Queens", "116": "Queens"}.get(z, "NY State")


def save_outputs(out, quiet=False):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pid = out["patient_id"]
    with open(f"{RESULTS_DIR}/{pid}.json", "w") as f:
        json.dump(out, f, indent=2)

    icon = {"PASS": "✅", "FAIL": "❌", "UNKNOWN": "❓"}
    f_ = out["funnel"]
    lines = [f"# Trial screening report — {pid}", "",
             f"**Patient:** {out['summary']}", "",
             f"**Funnel:** {f_['all_trials']} NYC recruiting trials → "
             f"{f_['after_prefilter']} after rules → {f_['screened_by_llm']} screened by AI", ""]
    if out["profile"].get("missing_information"):
        lines += ["**Missing from the summary:** " +
                  "; ".join(out["profile"]["missing_information"]), ""]
    lines += ["| Rank | Trial | Phase | Result | Pass / Fail / Unknown (important) |",
              "|---|---|---|---|---|"]
    for i, r in enumerate(out["results"], 1):
        lines.append(f"| {i} | [{r['nct_id']}]({r['url']}) {r['title'][:70]} | "
                     f"{'/'.join(r['phases']) or 'NA'} | {r['category']} | "
                     f"{r['n_pass']} / {r['n_fail']} / {r['n_unknown']} "
                     f"({r.get('n_unknown_important', r['n_unknown'])}) |")
    for i, r in enumerate(out["results"], 1):
        lines += ["", "---", f"## {i}. {r['nct_id']} — {r['title']}",
                  f"**{r['category']}** · fit {r['fit_score']}% · {r['url']}", "",
                  r.get("summary", ""), "", "**Checklist**", ""]
        for c in r.get("criteria", []):
            tag = ""
            if c.get("status") == "UNKNOWN":
                tag = " (routine)" if c.get("importance") == "routine" else " (important)"
            elif c.get("status") == "FAIL" and c.get("fail_confidence"):
                tag = f" ({c['fail_confidence'].lower()} confidence)"
            lines.append(f"- {icon.get(c.get('status'), '•')}{tag} *{c.get('type', '')}* — "
                         f"{c.get('criterion', '')}: {c.get('evidence', '')}")
        if r.get("open_questions"):
            lines += ["", "**Questions for the care team**", ""]
            lines += [f"- {q}" for q in r["open_questions"]]
        if r.get("model"):
            lines.append(f"_Screened with {r['model']}_")
        if r["nyc_sites"]:
            lines += ["", "**NYC sites**", ""]
            for s in r["nyc_sites"]:
                contact = ", ".join(x for x in [s.get("contact_name"), s.get("contact_phone"),
                                                s.get("contact_email")] if x)
                status = (s.get("site_status") or "status unknown").replace("_", " ").lower()
                lines.append(f"- {s['facility']} ({borough(s.get('zip'))}, {s.get('zip') or 'no zip'}; "
                             f"{status})" + (f" — {contact}" if contact else ""))
    with open(f"{RESULTS_DIR}/{pid}.md", "w") as f:
        f.write("\n".join(lines))
    if not quiet:
        print(f"  [6] saved {RESULTS_DIR}/{pid}.json and {RESULTS_DIR}/{pid}.md")


def print_summary(out):
    print(f"\n  RESULTS for {out['patient_id']}:")
    for i, r in enumerate(out["results"], 1):
        print(f"   {i:>2}. {r['nct_id']}  {r['category']:<33} "
              f"P/F/U={r['n_pass']}/{r['n_fail']}/{r['n_unknown']}  {r['title'][:45]}")


def main():
    global TRIALS_FILE, RESULTS_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", help="patient_id to run (default: all)")
    ap.add_argument("--top", type=int, default=TOP_N, help="trials to screen per patient")
    ap.add_argument("--trials", default=TRIALS_FILE, help="trial database (e.g. trials_v2.json)")
    ap.add_argument("--results", default=RESULTS_DIR, help="folder for results")
    ap.add_argument("--fresh", action="store_true", help="ignore saved results and start over")
    args = ap.parse_args()
    TRIALS_FILE, RESULTS_DIR = args.trials, args.results

    trials = json.load(open(TRIALS_FILE))
    patients = json.load(open(PATIENTS_FILE))
    if args.patient:
        patients = [p for p in patients if p["patient_id"] == args.patient]
        if not patients:
            raise SystemExit(f"No patient {args.patient} in {PATIENTS_FILE}")
    print(f"Model: {MODEL}   Trials: {TRIALS_FILE} ({len(trials)})   Results: {RESULTS_DIR}")

    for p in patients:
        try:
            out = screen_patient(p, trials, args.top, fresh=args.fresh)
        except DailyQuotaExceeded:
            raise SystemExit(1)
        print_summary(out)


if __name__ == "__main__":
    main()
