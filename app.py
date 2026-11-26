"""
app.py: coordinator interface for the Breast Cancer Clinical Trial Matcher.

In Colab:
  !pip install -q gradio google-genai
  import app; app.launch()          # prints a public link (valid ~72 hours)

Two tabs:
  Screen a patient   paste a de-identified summary -> live screening, ranked trial cards
  Saved results      browse the saved runs for P01-P05 (no API calls: safe for demos)
"""

import glob
import html
import json
import os
import re

import gradio as gr

import screener

TRIALS_FILE = os.environ.get("TRIALS_FILE", "trials_v2.json")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results_v4")
PATIENTS_FILE = "patients.json"

# ------------------------------------------------------------------ branding
BRAND = "Eligio"
TAGLINE = "The right trial, found faster."
SUBTITLE = "Breast cancer clinical trial pre-screening for New York City research coordinators"
CREDIT = "NYU Stern · TECH-GB.3332 · Taryn Samet &amp; Grace Kirkpatrick"
INK, ACCENT, SOFT = "#33245c", "#d9467a", "#f6f2fa"
LOGO = f"""<svg width='52' height='52' viewBox='0 0 52 52' role='img' aria-label='Eligio logo'>
  <rect x='0' y='0' width='52' height='52' rx='13' fill='{INK}'/>
  <circle cx='22' cy='22' r='12.5' fill='none' stroke='#fff' stroke-width='4'/>
  <line x1='31' y1='31' x2='41' y2='41' stroke='#fff' stroke-width='5' stroke-linecap='round'/>
  <polyline points='15.5,22.5 20.5,27.5 29,17' fill='none' stroke='{ACCENT}' stroke-width='4'
            stroke-linecap='round' stroke-linejoin='round'/>
</svg>"""
HEADER = f"""<style>
.tl-head{{display:flex;align-items:center;gap:16px;background:{SOFT};border-radius:14px;
  padding:16px 20px;margin-bottom:6px;border-left:6px solid {ACCENT}}}
.tl-name{{font-size:30px;font-weight:800;color:{INK};letter-spacing:-0.5px;line-height:1}}
.tl-name span{{color:{ACCENT}}}
.tl-tag{{font-size:17px;color:{INK};font-weight:600;margin-top:4px}}
.tl-sub{{font-size:13px;color:#5d5670;margin-top:2px}}
.tl-credit{{margin-left:auto;font-size:12px;color:#7a7390;text-align:right}}
.gradio-container, :root{{--color-accent:{INK} !important;--color-accent-soft:{SOFT} !important;
  --slider-color:{INK} !important;--button-primary-background-fill:{INK} !important;
  --button-primary-background-fill-hover:{ACCENT} !important;--button-primary-border-color:{INK} !important;
  --loader-color:{ACCENT} !important;--border-color-accent:{ACCENT} !important}}
button[role='tab'][aria-selected='true']{{color:{INK} !important;border-color:{ACCENT} !important}}
#tl-go{{background:{INK} !important;border-color:{INK} !important;color:#fff !important}}
#tl-go:hover{{background:{ACCENT} !important;border-color:{ACCENT} !important}}
</style>
<div class='tl-head'>{LOGO}
  <div><div class='tl-name' style='color:{INK}'>Elig<span style='color:{ACCENT}'>io</span></div>
  <div class='tl-tag'>{TAGLINE}</div><div class='tl-sub'>{SUBTITLE}</div></div>
  <div class='tl-credit'>{CREDIT}</div>
</div>"""

COLORS = {
    screener.ELIGIBLE: ("#1e7b34", "#e3f4e6"),
    screener.LIKELY_ELIGIBLE: ("#1d5fa8", "#e4eefa"),
    screener.LIKELY_INELIGIBLE: ("#9a5b00", "#fdf0dc"),
    screener.NOT_ELIGIBLE: ("#a12a2a", "#fbe5e5"),
    screener.ERROR_CATEGORY: ("#555", "#eee"),
}
CSS = """
.tm-card{border:1px solid #ddd;border-radius:10px;padding:14px 16px;margin:10px 0;background:#fff;color:#222}
.tm-badge{display:inline-block;padding:3px 10px;border-radius:12px;font-weight:600;font-size:13px}
.tm-title{font-size:15px;font-weight:600;margin:6px 0}
.tm-meta{color:#666;font-size:13px}
.tm-crit{font-size:13px;margin:3px 0}
.tm-sum td,.tm-sum th{padding:4px 10px;border-bottom:1px solid #eee;text-align:left;color:#222}
.tm-note{background:#fff8e1;border:1px solid #f0d68a;border-radius:8px;padding:8px 12px;color:#5a4500;font-size:13px}
details summary{cursor:pointer;color:#1d5fa8;font-size:13px;margin-top:6px}
"""
DISCLAIMER = ("<div class='tm-note'><b>Decision support only.</b> Results are AI pre-screening of "
              "public eligibility criteria against a de-identified summary. The coordinator and the "
              "trial site make every eligibility decision. Prototype built on synthetic patients.</div>")


def esc(x):
    return html.escape(str(x or ""))


def badge(cat):
    fg, bg = COLORS.get(cat, ("#555", "#eee"))
    return f"<span class='tm-badge' style='color:{fg};background:{bg}'>{esc(cat)}</span>"


def card(rank, r):
    icon = {"PASS": "✅", "FAIL": "❌", "UNKNOWN": "❓"}
    # failures first, then unknowns (important before routine), then criteria that were met
    def order(c):
        st = c.get("status")
        if st == "FAIL":
            return 0
        if st == "UNKNOWN":
            return 1 if c.get("importance") != "routine" else 2
        return 3
    crit_html = ""
    for c in sorted(r.get("criteria", []), key=order):
        tag = ""
        if c.get("status") == "UNKNOWN":
            tag = " <i>(routine)</i>" if c.get("importance") == "routine" else " <b>(important)</b>"
        name = esc(c.get("criterion"))
        name = name if c.get("status") == "PASS" else f"<b>{name}</b>"
        crit_html += (f"<div class='tm-crit'>{icon.get(c.get('status'), '•')}{tag} "
                      f"{name}: {esc(c.get('evidence'))}</div>")
    qs = "".join(f"<li>{esc(q)}</li>" for q in r.get("open_questions", []))
    sites = "".join(
        f"<li>{esc(s.get('facility'))} ({esc(screener.borough(s.get('zip')))}; "
        f"{esc((s.get('site_status') or '').replace('_', ' ').lower())})"
        + (f" · {esc(', '.join(x for x in [s.get('contact_name'), s.get('contact_phone'), s.get('contact_email')] if x))}"
           if any([s.get('contact_name'), s.get('contact_phone'), s.get('contact_email')]) else "")
        + "</li>" for s in r.get("nyc_sites", [])[:5])
    cc = r.get("central_contact") or {}
    central = ", ".join(x for x in [cc.get("name"), cc.get("phone"), cc.get("email")] if x)
    return f"""
<div class='tm-card'>
  <div>{badge(r['category'])} <span class='tm-meta'>#{rank} · {esc('/'.join(r.get('phases', [])) or 'NA')} ·
   {r['n_pass']} met · {r['n_fail']} not met · {r.get('n_unknown_important', r['n_unknown'])} important unknown</span></div>
  <div class='tm-title'><a href='{esc(r['url'])}' target='_blank'>{esc(r['nct_id'])}</a> · {esc(r['title'])}</div>
  <div>{esc(r.get('summary'))}</div>
  {f"<div style='margin-top:8px'><b>Questions for the care team</b><ul>{qs}</ul></div>" if qs else ""}
  <details><summary>Criterion-by-criterion checklist</summary>{crit_html}</details>
  <details><summary>NYC sites and contacts</summary><ul>{sites}</ul>
   {f"<div class='tm-meta'>Central contact: {esc(central)}</div>" if central else ""}</details>
</div>"""


def render(out, status=""):
    counts = {}
    for r in out["results"]:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    rows = "".join(f"<tr><td>{badge(c)}</td><td>{counts.get(c, 0)}</td></tr>"
                   for c in screener.CATEGORY_ORDER if counts.get(c))
    f = out.get("funnel", {})
    missing = out.get("profile", {}).get("missing_information") or []
    head = f"""<style>{CSS}</style>
{DISCLAIMER}
<p class='tm-meta' style='font-size:13px'>{esc(status)}</p>
<p><b>Searched:</b> {f.get('all_trials', '?')} recruiting breast cancer treatment trials with an open NYC site →
{f.get('after_prefilter', '?')} after rules (age, sex, treatment trials) → {f.get('screened_by_llm', '?')} checked in detail.</p>
<table class='tm-sum'><tr><th>Result</th><th>Trials</th></tr>{rows}</table>
{f"<p><b>Information missing from the summary:</b> {esc('; '.join(missing))}</p>" if missing else ""}"""
    return head + "".join(card(i, r) for i, r in enumerate(out["results"], 1))


# ------------------------------------------------------------------ key-facts form
# The form is optional. Coordinators can use the summary only, the form only, or both.
# When both are used, the SUMMARY WINS: form values only fill gaps the summary leaves.
SETTINGS = ["Early, before surgery", "Early, after surgery", "Locally advanced / unresectable", "Metastatic"]
RECEPTOR = ["Positive", "Low (1-10%)", "Negative", "Unknown"]
PR_OPTS = ["Positive", "Negative", "Unknown"]
HER2_OPTS = ["0", "1+ (HER2-low)", "2+ ISH negative (HER2-low)", "2+ ISH positive", "3+", "Unknown"]
LINES = ["0", "1", "2", "3+"]
ECOG = ["0", "1", "2", "3", "4"]
BRAIN = ["None", "Treated, stable", "Active", "Not imaged"]
TREATMENTS = ["Endocrine therapy", "CDK4/6 inhibitor", "Chemotherapy", "Anti-HER2 antibody",
              "T-DXd or other ADC", "HER2 TKI (e.g. tucatinib)", "Immunotherapy (PD-1/PD-L1)",
              "PARP inhibitor", "PI3K/AKT/mTOR inhibitor"]
MARKER_OPTS = ["Mutation / positive", "Tested negative", "Not tested"]
MARKERS = [("brca", "BRCA1/2 or PALB2"), ("pik3ca", "PIK3CA"), ("esr1", "ESR1"), ("pdl1", "PD-L1 (TNBC)")]
FIELDS = ["age", "sex", "meno", "setting", "er", "pr", "her2", "lines", "ecog", "brain",
          "treatments", "brca", "pik3ca", "esr1", "pdl1", "other"]
EMPTY = [None, None, None, None, None, None, None, None, None, None, [], None, None, None, None, ""]

FORM_FILL_PROMPT = f"""You fill in a structured form from a breast cancer patient summary for clinical
trial screening. Use only what the summary states; use null when it does not say. Never guess.
Return JSON with exactly these keys, each value exactly one of the allowed options or null:
  "age": integer
  "sex": "Female" | "Male"
  "meno": "Pre" | "Post" | "Unknown"
  "setting": {SETTINGS}
  "er": {RECEPTOR}      (Low means 1-10% staining)
  "pr": {PR_OPTS}
  "her2": {HER2_OPTS}
  "lines": {LINES}      (number of prior treatment lines for metastatic disease)
  "ecog": {ECOG}
  "brain": {BRAIN}      ("Not imaged" if brain imaging is not mentioned)
  "treatments": a list with any of {TREATMENTS} that the patient has received, in any setting
  "brca": {MARKER_OPTS}   (germline or tumor BRCA1, BRCA2 or PALB2)
  "pik3ca": {MARKER_OPTS}
  "esr1": {MARKER_OPTS}
  "pdl1": {MARKER_OPTS}   (Mutation / positive means PD-L1 positive, e.g. CPS >= 10)
  "other": short text listing other named genomic alterations, or ""
"""


def _form(values):
    return dict(zip(FIELDS, values))


def form_is_empty(f):
    return all(v in (None, "", []) for v in f.values())


def form_profile(f):
    """Translate the form into the screener's patient profile (only what was filled in)."""
    prof = {}
    if f.get("age") not in (None, ""):
        prof["age"] = int(f["age"])
    if f.get("sex"):
        prof["sex"] = f["sex"].upper()
    if f.get("meno") in ("Pre", "Post"):
        prof["menopausal_status"] = "premenopausal" if f["meno"] == "Pre" else "postmenopausal"
    if f.get("setting"):
        prof["disease_setting"] = "early" if f["setting"].startswith("Early") else "metastatic"
    er, pr, her2 = f.get("er"), f.get("pr"), f.get("her2")
    hr_pos = er in ("Positive", "Low (1-10%)") or pr == "Positive"
    if her2 in ("2+ ISH positive", "3+"):
        prof["subtype"] = "HER2+"
    elif hr_pos:
        prof["subtype"] = "HR+/HER2-"
    elif er == "Negative" and pr == "Negative" and her2 in ("0", "1+ (HER2-low)", "2+ ISH negative (HER2-low)"):
        prof["subtype"] = "TNBC"
    if her2 and her2 != "Unknown":
        prof["her2_low"] = her2 in ("1+ (HER2-low)", "2+ ISH negative (HER2-low)")
    if f.get("lines"):
        prof["lines_of_therapy_metastatic"] = 3 if f["lines"] == "3+" else int(f["lines"])
    if f.get("ecog"):
        prof["ecog"] = int(f["ecog"])
    if f.get("brain"):
        prof["brain_metastases"] = {"None": "none", "Treated, stable": "treated_stable",
                                    "Active": "active", "Not imaged": "unknown"}[f["brain"]]
    if f.get("treatments"):
        prof["prior_therapies"] = list(f["treatments"])
    markers = []
    for key, name in MARKERS:
        if f.get(key) == "Mutation / positive":
            markers.append(f"{name} positive")
        elif f.get(key) == "Tested negative":
            markers.append(f"{name} tested negative")
    if (f.get("other") or "").strip():
        markers.append(f["other"].strip())
    if markers:
        prof["biomarkers"] = markers
    return prof


def form_text(f):
    """Plain-language version of the form for the AI's detailed check."""
    parts = []
    if f.get("age") not in (None, "") or f.get("sex"):
        parts.append(f"{f.get('age') or 'Age not given'}-year-old {(f.get('sex') or 'patient').lower()}"
                     + (f", {f['meno'].lower()}menopausal" if f.get("meno") in ("Pre", "Post") else "") + ".")
    if f.get("setting"):
        parts.append(f"Disease setting: {f['setting']}.")
    rec = [f"ER {f['er']}" if f.get("er") else "", f"PR {f['pr']}" if f.get("pr") else "",
           f"HER2 {f['her2']}" if f.get("her2") else ""]
    if any(rec):
        parts.append("Receptors: " + ", ".join(x for x in rec if x) + ".")
    if f.get("lines"):
        parts.append(f"Prior treatment lines for metastatic disease: {f['lines']}.")
    if f.get("treatments"):
        parts.append("Prior treatment received: " + ", ".join(f["treatments"]) + ".")
    if f.get("ecog"):
        parts.append(f"ECOG performance status {f['ecog']}.")
    if f.get("brain"):
        parts.append(f"Brain metastases: {f['brain']}.")
    for key, name in MARKERS:
        if f.get(key):
            parts.append(f"{name}: {f[key].lower()}.")
    if (f.get("other") or "").strip():
        parts.append(f"Other genomic alterations: {f['other'].strip()}.")
    return " ".join(parts)


def merge_profiles(from_summary, from_form):
    """Summary wins. The form fills only what the summary left empty. Returns (profile, conflicts)."""
    merged, conflicts = dict(from_summary), []
    labels = {"age": "age", "sex": "sex", "subtype": "subtype", "disease_setting": "disease setting",
              "ecog": "ECOG", "brain_metastases": "brain metastases",
              "lines_of_therapy_metastatic": "prior lines for metastatic disease"}
    for k, v in from_form.items():
        sv = merged.get(k)
        if sv in (None, "", [], "unknown"):
            merged[k] = v
        elif k in labels and str(sv).lower() != str(v).lower():
            conflicts.append(f"{labels[k]} (form: {v}; summary: {sv})")
        elif k == "biomarkers":
            have = " ".join(sv).upper()
            merged[k] = list(sv) + [b for b in v if b.split()[0].upper() not in have]
    return merged, conflicts


def fill_from_summary(summary):
    """'Fill from summary' button: the AI pre-fills the form; the coordinator checks it."""
    if not summary or len(summary.strip()) < 40:
        return [gr.update() for _ in FIELDS] + ["<p class='tm-meta'>Paste a summary first, then click Fill from summary.</p>"]
    keep = [gr.update() for _ in FIELDS]          # leaves whatever is in the form untouched
    try:
        d = screener.call_llm(FORM_FILL_PROMPT, f"PATIENT SUMMARY:\n{summary}")
    except screener.DailyQuotaExceeded:
        return keep + ["<p><b>The free daily AI limit has been reached.</b> The form was not changed; "
                       "you can still fill it in by hand.</p>"]
    except Exception as e:
        log_error("fill_from_summary: AI call", e)
        return keep + [f"<p><b>Could not fill the form</b> (the form was not changed): {esc(str(e)[:200])}</p>"]
    if isinstance(d, list) and d and isinstance(d[0], dict):
        d = d[0]
    if not isinstance(d, dict):
        log_error("fill_from_summary: unexpected answer", TypeError(repr(d)[:300]))
        return keep + ["<p><b>The AI returned an answer Eligio couldn't read.</b> The form was not "
                       "changed. Please try again, or fill it in by hand.</p>"]
    try:
        out = [_coerce_field(k, d.get(k)) for k in FIELDS]
    except Exception as e:                      # should never happen; never crash the form
        log_error("fill_from_summary: coercion", e)
        return keep + ["<p><b>Could not fill the form</b> (the form was not changed). Please fill it in by hand.</p>"]
    n = sum(1 for v in out if v not in (None, "", []))
    return out + [f"<p class='tm-meta'>Filled {n} field(s) from the summary. Fields the summary doesn't "
                  "mention are left blank. Please check everything, and correct anything wrong.</p>"]


FILL_ALLOWED = {"sex": ["Female", "Male"], "meno": ["Pre", "Post", "Unknown"], "setting": SETTINGS,
                "er": RECEPTOR, "pr": PR_OPTS, "her2": HER2_OPTS, "lines": LINES, "ecog": ECOG,
                "brain": BRAIN, "brca": MARKER_OPTS, "pik3ca": MARKER_OPTS, "esr1": MARKER_OPTS,
                "pdl1": MARKER_OPTS}


def _match_option(v, options):
    """Exact or case-insensitive match to one allowed option, else None."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        v = str(int(v)) if float(v).is_integer() else str(v)
    if not isinstance(v, str):
        return None
    v = v.strip()
    for o in options:
        if v == o or v.lower() == o.lower():
            return o
    return None


def _coerce_field(k, v):
    """Turn one AI value into something the form component accepts (or blank)."""
    if isinstance(v, dict):                     # e.g. {"value": "Positive"}
        v = v.get("value")
    if k == "age":
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            age = int(v)
        else:
            m = re.match(r"\s*(\d{1,3})(\.0+)?\b", str(v or ""))
            age = int(m.group(1)) if m else None
        return age if age is not None and 18 <= age <= 110 else None
    if k == "treatments":
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list):
            return []
        picked = [_match_option(t, TREATMENTS) for t in v]
        return [t for t in TREATMENTS if t in picked]      # form order, no duplicates
    if k == "other":
        if isinstance(v, list):
            v = ", ".join(str(x) for x in v if x)
        return v.strip()[:300] if isinstance(v, str) else ""
    if k == "lines" and isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 3:
        return "3+"
    return _match_option(v, FILL_ALLOWED.get(k, []))


def log_error(where, e):
    """Print a full traceback to the Render log so failures can be diagnosed."""
    import traceback
    print(f"[Eligio error] {where}: {type(e).__name__}: {e}", flush=True)
    traceback.print_exception(e)


def clear_form():
    return list(EMPTY) + [""]


# ------------------------------------------------------------------ live screening
def screen_live(summary, top_n, *form_values):
    summary = (summary or "").strip()
    f = _form(form_values)
    has_summary, has_form = len(summary) >= 40, not form_is_empty(f)
    if not has_summary and not has_form:
        yield ("<p>Paste a patient summary, fill in the key-facts form, or both "
               "(or load an example patient).</p>")
        return
    trials = json.load(open(TRIALS_FILE))
    try:
        mode = ("summary + form" if has_summary and has_form else "summary only" if has_summary else "form only")
        print(f"[eligio] new screening request ({mode})", flush=True)
        conflicts = []
        if has_summary:
            yield "<p>Step 1 of 3 · Reading the patient summary…</p>"
            profile = None
            for attempt in range(3):          # the parser can also return malformed JSON
                try:
                    profile = screener.call_llm(screener.PARSER_PROMPT, f"PATIENT SUMMARY:\n{summary}")
                    if isinstance(profile, dict):
                        break
                except screener.DailyQuotaExceeded:
                    raise
                except ValueError as e:
                    print(f"[eligio] parser retry {attempt + 1}: {e}", flush=True)
            if not isinstance(profile, dict):
                raise RuntimeError("The AI could not read this summary. Please try again.")
            if has_form:
                profile, conflicts = merge_profiles(profile, form_profile(f))
        else:
            profile = form_profile(f)
        # text the AI reads for the detailed check
        if has_summary and has_form:
            text = (summary + "\n\nAdditional facts entered by the coordinator (use only where the "
                    "summary above is silent; the summary takes precedence): " + form_text(f))
        else:
            text = summary if has_summary else form_text(f)

        kept, removed = screener.prefilter(trials, profile)
        top, _ = screener.rank_trials(kept, profile, int(top_n))
        note = f"Used: {mode}."
        if conflicts:
            note += " The form and the summary disagree on " + "; ".join(conflicts) + ". Eligio used the summary."
        results = []
        for i, (rel, why, t) in enumerate(top, 1):
            partial = {"results": sorted(results, key=lambda r: screener.CATEGORY_ORDER[r["category"]]),
                       "profile": profile,
                       "funnel": {"all_trials": len(trials), "after_prefilter": len(kept),
                                  "screened_by_llm": len(top)}}
            yield render(partial, f"{note} Step 3 of 3 · Checking trial {i} of {len(top)} ({t['nct_id']})…")
            try:
                r = screener.screen_one(text, t)
            except screener.DailyQuotaExceeded:
                raise
            except Exception as e:
                print(f"[eligio] trial {t['nct_id']} could not be assessed: {e}", flush=True)
                r = {"summary": f"Could not assess this trial: {e}", "criteria": [], "open_questions": []}
            r.update(screener.score_checklist(r))
            r.update({"nct_id": t["nct_id"], "title": t["title"], "url": t["url"],
                      "phases": t.get("phases", []), "nyc_sites": t.get("nyc_sites", []),
                      "central_contact": t.get("central_contact", {})})
            results.append(r)
        results.sort(key=lambda r: (screener.CATEGORY_ORDER[r["category"]],
                                    r.get("n_unknown_important", 0), -r["fit_score"]))
        out = {"results": results, "profile": profile,
               "funnel": {"all_trials": len(trials), "after_prefilter": len(kept),
                          "screened_by_llm": len(top)}}
        yield render(out, f"{note} Done · model {screener.MODEL}")
    except screener.DailyQuotaExceeded:
        yield ("<p><b>The free daily limit for the AI model has been reached.</b> Use the "
               "<i>Saved results</i> tab, or try again tomorrow.</p>")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[eligio] screening failed: {e}", flush=True)
        yield (f"<p><b>Something went wrong:</b> {esc(e)}</p><p class='tm-meta'>If the site was "
               f"asleep, it may still be starting: wait 30 seconds and click the button again.</p>")


# ------------------------------------------------------------------ saved results
def saved_ids():
    return sorted(os.path.basename(f)[:-5] for f in glob.glob(f"{RESULTS_DIR}/P*.json"))


def show_saved(pid):
    if not pid:
        return "<p>No saved results found.</p>"
    d = json.load(open(f"{RESULTS_DIR}/{pid}.json"))
    when = d.get("experiment", {}).get("timestamp_utc", "")[:16].replace("T", " ")
    return (f"<p><b>{esc(pid)}:</b> {esc(d.get('label'))}</p>"
            f"<details><summary>Patient summary</summary><p>{esc(d.get('summary'))}</p></details>"
            + render(d, f"Saved run {when} UTC · model {', '.join(d.get('experiment', {}).get('models_actually_used', []))}"))


def example_text(pid):
    for p in json.load(open(PATIENTS_FILE)):
        if p["patient_id"] == pid:
            return p["summary"]
    return ""


def build():
    pids = [p["patient_id"] for p in json.load(open(PATIENTS_FILE))]
    with gr.Blocks(title=f"{BRAND} · {TAGLINE}") as demo:
        gr.HTML(HEADER)
        with gr.Tab("Screen a patient"):
            with gr.Row():
                with gr.Column(scale=2):
                    summary = gr.Textbox(label="De-identified patient summary", lines=7,
                                         placeholder="Age, sex, diagnosis, ER/PR/HER2, stage, biomarkers, "
                                                     "prior treatments, performance status, labs…")
                    with gr.Accordion("Key patient facts (optional)", open=True):
                        gr.HTML("<div style='font-size:13px;color:#5d5670;margin:-4px 0 6px'>Use the "
                                "summary, this form, or both. <b>*</b> marks the facts that matter most "
                                "for matching. If the form and the summary disagree, Eligio uses the "
                                "summary.</div>")
                        with gr.Row():
                            fill = gr.Button("✨ Fill from summary", size="sm")
                            clear = gr.Button("Clear form", size="sm")
                        fill_msg = gr.HTML()
                        with gr.Row():
                            age = gr.Number(label="Age *", precision=0, value=None, minimum=0, maximum=120)
                            sex = gr.Radio(["Female", "Male"], label="Sex *", value=None)
                            meno = gr.Radio(["Pre", "Post", "Unknown"], label="Menopausal status", value=None)
                        with gr.Row():
                            setting = gr.Dropdown(SETTINGS, label="Disease setting *", value=None)
                            er = gr.Dropdown(RECEPTOR, label="ER *", value=None)
                            pr = gr.Dropdown(PR_OPTS, label="PR *", value=None)
                            her2 = gr.Dropdown(HER2_OPTS, label="HER2 *", value=None)
                        with gr.Row():
                            lines = gr.Dropdown(LINES, label="Prior lines for metastatic disease", value=None)
                            ecog = gr.Dropdown(ECOG, label="ECOG", value=None)
                            brain = gr.Dropdown(BRAIN, label="Brain metastases", value=None)
                        treatments = gr.CheckboxGroup(TREATMENTS, label="Prior treatment received (any setting)", value=[])
                        gr.HTML("<div style='font-size:14px;font-weight:600;margin-top:6px'>Biomarkers</div>")
                        with gr.Row():
                            brca = gr.Dropdown(MARKER_OPTS, label="BRCA1/2 or PALB2", value=None)
                            pik3ca = gr.Dropdown(MARKER_OPTS, label="PIK3CA", value=None)
                            esr1 = gr.Dropdown(MARKER_OPTS, label="ESR1", value=None)
                            pdl1 = gr.Dropdown(MARKER_OPTS, label="PD-L1 (TNBC)", value=None)
                        other = gr.Textbox(label="Other alterations (e.g. TP53 Y220C, AKT1, PTEN loss)", value="")
                with gr.Column(scale=1):
                    ex = gr.Dropdown(pids, value=None, label="Or load a synthetic example patient")
                    top_n = gr.Slider(3, 15, value=8, step=1, label="Trials to check in detail")
                    go = gr.Button("Find matching trials", variant="primary", elem_id="tl-go")
                    gr.Markdown("<small>About 10 seconds per trial checked.</small>")
            live = gr.HTML()
            form = [age, sex, meno, setting, er, pr, her2, lines, ecog, brain, treatments,
                    brca, pik3ca, esr1, pdl1, other]
            # loading an example fills the summary and clears the form
            ex.change(lambda pid: [example_text(pid)] + clear_form(), ex, [summary] + form + [fill_msg])
            fill.click(fill_from_summary, summary, form + [fill_msg])
            clear.click(clear_form, None, form + [fill_msg])
            go.click(screen_live, [summary, top_n] + form, live)
        with gr.Tab("Saved results (demo)"):
            ids = saved_ids()
            pick = gr.Dropdown(ids, value=ids[0] if ids else None, label="Synthetic patient")
            saved = gr.HTML(show_saved(ids[0]) if ids else "<p>No saved results yet.</p>")
            pick.change(show_saved, pick, saved)
    return demo


def launch():
    """Render: permanent onrender.com URL. Colab: public share link (~72 h).
    Set APP_PASSWORD (a Space secret) to require a login: user 'coordinator'."""
    pw = os.environ.get("APP_PASSWORD")
    auth = ("coordinator", pw) if pw else None
    if os.environ.get("RENDER") and os.environ.get("PORT"):   # Render sets both
        build().queue().launch(server_name="0.0.0.0", server_port=int(os.environ["PORT"]),
                               auth=auth, share=False, show_error=True)
    elif os.environ.get("SPACE_ID"):    # Hugging Face Spaces
        build().queue().launch(auth=auth)
    else:                               # Colab: temporary public share link
        build().queue().launch(share=True, debug=False, auth=auth, show_error=True)


if __name__ == "__main__":
    launch()
