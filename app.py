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
<p class='tm-meta'>{esc(status)}</p>
<p><b>Searched:</b> {f.get('all_trials', '?')} recruiting breast cancer treatment trials with an open NYC site →
{f.get('after_prefilter', '?')} after rules (age, sex, treatment trials) → {f.get('screened_by_llm', '?')} checked in detail.</p>
<table class='tm-sum'><tr><th>Result</th><th>Trials</th></tr>{rows}</table>
{f"<p><b>Information missing from the summary:</b> {esc('; '.join(missing))}</p>" if missing else ""}"""
    return head + "".join(card(i, r) for i, r in enumerate(out["results"], 1))


# ------------------------------------------------------------------ live screening
def screen_live(summary, top_n):
    if not summary or len(summary.strip()) < 40:
        yield ("<p>Please paste a patient summary (a few sentences: diagnosis, receptors, "
               "stage, biomarkers, prior treatments, performance status, labs), or load an "
               "example patient.</p>")
        return
    trials = json.load(open(TRIALS_FILE))
    record = {"patient_id": "UI", "summary": summary.strip()}
    try:
        yield "<p>Step 1 of 3 · Reading the patient summary…</p>"
        profile = screener.call_llm(screener.PARSER_PROMPT, f"PATIENT SUMMARY:\n{summary}")
        kept, removed = screener.prefilter(trials, profile)
        top, _ = screener.rank_trials(kept, profile, int(top_n))
        results = []
        for i, (rel, why, t) in enumerate(top, 1):
            partial = {"results": sorted(results, key=lambda r: screener.CATEGORY_ORDER[r["category"]]),
                       "profile": profile,
                       "funnel": {"all_trials": len(trials), "after_prefilter": len(kept),
                                  "screened_by_llm": len(top)}}
            yield render(partial, f"Step 3 of 3 · Checking trial {i} of {len(top)} ({t['nct_id']})…")
            try:
                r = screener.validate_screening_response(screener.call_llm(
                    screener.SCREENER_PROMPT,
                    f"PATIENT SUMMARY:\n{summary}\n\n---\n\n{screener.trial_text(t)}"))
            except screener.DailyQuotaExceeded:
                raise
            except Exception as e:
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
        yield render(out, f"Done · model {screener.MODEL}")
    except screener.DailyQuotaExceeded:
        yield ("<p><b>The free daily limit for the AI model has been reached.</b> Use the "
               "<i>Saved results</i> tab, or try again tomorrow.</p>")
    except Exception as e:
        yield f"<p><b>Something went wrong:</b> {esc(e)}</p>"


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


def example_from_event(evt: gr.SelectData):
    return example_text(evt.value)


def build():
    pids = [p["patient_id"] for p in json.load(open(PATIENTS_FILE))]
    with gr.Blocks(title=f"{BRAND} · {TAGLINE}") as demo:
        gr.HTML(HEADER)
        with gr.Tab("Screen a patient"):
            with gr.Row():
                with gr.Column(scale=2):
                    summary = gr.Textbox(label="De-identified patient summary", lines=9,
                                         placeholder="Age, sex, diagnosis, ER/PR/HER2, stage, biomarkers, "
                                                     "prior treatments, performance status, labs…")
                with gr.Column(scale=1):
                    ex = gr.Dropdown(pids, label="Or load a synthetic example patient")
                    top_n = gr.Slider(3, 15, value=8, step=1, label="Trials to check in detail")
                    go = gr.Button("Find matching trials", variant="primary", elem_id="tl-go")
                    gr.Markdown("<small>About 10 seconds per trial checked.</small>")
            live = gr.HTML()
            ex.change(example_text, ex, summary)
            ex.select(example_from_event, None, summary)
            go.click(screen_live, [summary, top_n], live)
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
                               auth=auth, share=False)
    elif os.environ.get("SPACE_ID"):    # Hugging Face Spaces
        build().queue().launch(auth=auth)
    else:                               # Colab: temporary public share link
        build().queue().launch(share=True, debug=False, auth=auth)


if __name__ == "__main__":
    launch()
