"""
fetch_trials.py — Trial retrieval module for the breast cancer trial matcher.

Pulls recruiting, interventional breast cancer trials with at least one site
near NYC from the ClinicalTrials.gov v2 API, cleans them into a compact
record per trial, and saves them to a local cache (JSON + CSV).

Run:  python fetch_trials.py
Needs: pip install requests   (already installed in Google Colab)
"""

import csv
import json
import math
import re
import time

import requests

# ---------------------------------------------------------------- settings
API_URL = "https://clinicaltrials.gov/api/v2/studies"

CONDITION = "breast cancer"          # API expands synonyms (e.g. breast neoplasm)
CENTER_LAT, CENTER_LON = 40.7128, -74.0060   # Lower Manhattan
RADIUS_MI = 15                       # 15 mi covers the five boroughs (+ a bit of NJ)
ALLOWED_STATES = {"New York"}        # set to None to keep nearby NJ sites too

# Primary-purpose filter. Leave False on the first run to see what it would
# remove (see the printed counts / CSV), then flip to True if you agree.
TREATMENT_ONLY = False

OUT_JSON = "trials.json"
OUT_CSV = "trials_summary.csv"

# Only request the parts of each record we use (keeps responses small).
FIELDS = ",".join([
    "protocolSection.identificationModule",
    "protocolSection.statusModule",
    "protocolSection.designModule",
    "protocolSection.conditionsModule",
    "protocolSection.armsInterventionsModule",
    "protocolSection.eligibilityModule",
    "protocolSection.contactsLocationsModule",
])


# ---------------------------------------------------------------- 1. fetch
def fetch_all_studies():
    """Page through the API and return the raw study records."""
    params = {
        "query.cond": CONDITION,
        "filter.overallStatus": "RECRUITING",
        "filter.geo": f"distance({CENTER_LAT},{CENTER_LON},{RADIUS_MI}mi)",
        "filter.advanced": "AREA[StudyType]INTERVENTIONAL",
        "fields": FIELDS,
        "pageSize": 100,
        "countTotal": "true",
        "format": "json",
    }
    studies, page = [], 1
    while True:
        resp = requests.get(API_URL, params=params, timeout=30)
        if resp.status_code == 400 and "fields" in params:
            # If the API ever rejects the field list, fall back to full records.
            print("API rejected field list; retrying with full records.")
            params.pop("fields")
            continue
        resp.raise_for_status()
        data = resp.json()

        if page == 1:
            print(f"API reports {data.get('totalCount', '?')} matching trials.")
        studies.extend(data.get("studies", []))
        print(f"  page {page}: {len(studies)} trials so far")

        token = data.get("nextPageToken")
        if not token:
            break
        params["pageToken"] = token
        page += 1
        time.sleep(0.3)  # be polite to the API
    return studies


# ---------------------------------------------------------------- 2. clean
def miles_between(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles (haversine)."""
    r = 3958.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def parse_age(text):
    """'18 Years' -> 18.0 ; '6 Months' -> 0.5 ; missing -> None."""
    if not text:
        return None
    m = re.match(r"(\d+(?:\.\d+)?)\s*(year|month|week|day)", text, re.I)
    if not m:
        return None
    n, unit = float(m.group(1)), m.group(2).lower()
    return {"year": n, "month": n / 12, "week": n / 52, "day": n / 365}[unit]


def split_criteria(text):
    """Split free-text eligibility into inclusion and exclusion sections."""
    if not text:
        return "", ""
    parts = re.split(r"\n\s*exclusion criteria\s*:?", text, maxsplit=1, flags=re.I)
    inclusion = re.sub(r"^\s*inclusion criteria\s*:?", "", parts[0], flags=re.I).strip()
    exclusion = parts[1].strip() if len(parts) > 1 else ""
    return inclusion, exclusion



def nyc_sites(locations):
    """Keep only actively recruiting sites within the radius, nearest first."""
    sites = []

    for loc in locations or []:
        # Only include locations actively recruiting patients.
        if loc.get("status") != "RECRUITING":
            continue

        geo = loc.get("geoPoint") or {}
        if "lat" not in geo or "lon" not in geo:
            continue

        dist = miles_between(CENTER_LAT, CENTER_LON, geo["lat"], geo["lon"])
        if dist > RADIUS_MI:
            continue

        if ALLOWED_STATES and loc.get("state") not in ALLOWED_STATES:
            continue

        contact = (loc.get("contacts") or [{}])[0]

        sites.append({
            "facility": loc.get("facility"),
            "city": loc.get("city"),
            "state": loc.get("state"),
            "zip": loc.get("zip"),
            "site_status": loc.get("status"),
            "distance_mi": round(dist, 1),
            "contact_name": contact.get("name"),
            "contact_phone": contact.get("phone"),
            "contact_email": contact.get("email"),
        })

    return sorted(sites, key=lambda s: s["distance_mi"])



def clean_study(raw):
    """Turn one raw API record into the compact record our app uses."""
    ps = raw.get("protocolSection", {})
    ident = ps.get("identificationModule", {})
    design = ps.get("designModule", {})
    elig = ps.get("eligibilityModule", {})
    arms = ps.get("armsInterventionsModule", {})
    contacts = ps.get("contactsLocationsModule", {})

    criteria_text = elig.get("eligibilityCriteria", "")
    inclusion, exclusion = split_criteria(criteria_text)
    central = (contacts.get("centralContacts") or [{}])[0]
    conditions = ps.get("conditionsModule", {}).get("conditions", [])
    title = ident.get("briefTitle") or ""

    return {
        "nct_id": ident.get("nctId"),
        "title": title,
        "official_title": ident.get("officialTitle"),
        "status": ps.get("statusModule", {}).get("overallStatus"),
        "phases": design.get("phases", []),
        # TREATMENT, SUPPORTIVE_CARE, PREVENTION, DIAGNOSTIC, SCREENING, ...
        "primary_purpose": (design.get("designInfo") or {}).get("primaryPurpose", "UNKNOWN"),
        "conditions": conditions,
        # Flag (not a filter): does "breast" appear in the conditions or title?
        # False usually means a basket solid-tumor trial OR an off-topic match.
        "breast_mentioned": any("breast" in s.lower() for s in conditions + [title]),
        "interventions": [
            f"{i.get('type', '')}: {i.get('name', '')}".strip(": ")
            for i in arms.get("interventions", [])
        ],
        # structured fields -> used by the cheap pre-filter
        "sex": elig.get("sex"),                       # ALL / FEMALE / MALE
        "min_age": parse_age(elig.get("minimumAge")),
        "max_age": parse_age(elig.get("maximumAge")),
        # free text -> sent to the LLM screener
        "inclusion_criteria": inclusion,
        "exclusion_criteria": exclusion,
        "criteria_raw": criteria_text,
        # contacts -> shown on the results card
        "central_contact": {
            "name": central.get("name"),
            "phone": central.get("phone"),
            "email": central.get("email"),
        },
        "nyc_sites": nyc_sites(contacts.get("locations")),
        "url": f"https://clinicaltrials.gov/study/{ident.get('nctId')}",
    }


# ---------------------------------------------------------------- 3. save
def save(trials):
    with open(OUT_JSON, "w") as f:
        json.dump(trials, f, indent=2)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["nct_id", "title", "phases", "primary_purpose",
                    "breast_mentioned", "sex", "min_age", "max_age",
                    "n_nyc_sites", "nearest_site", "criteria_chars"])
        for t in trials:
            w.writerow([
                t["nct_id"], t["title"], "/".join(t["phases"]),
                t["primary_purpose"], t["breast_mentioned"], t["sex"],
                t["min_age"], t["max_age"], len(t["nyc_sites"]),
                t["nyc_sites"][0]["facility"] if t["nyc_sites"] else "",
                len(t["criteria_raw"]),
            ])


def main():
    raw = fetch_all_studies()
    trials = [clean_study(s) for s in raw]

    # Drop trials with no qualifying NYC site or no eligibility text.
    kept = [t for t in trials if t["nyc_sites"] and t["criteria_raw"]]
    print(f"\nFetched {len(trials)} trials; kept {len(kept)} with NYC sites + criteria.")

    # Primary-purpose breakdown, so you can judge the filter before applying it.
    purpose_counts = {}
    for t in kept:
        purpose_counts[t["primary_purpose"]] = purpose_counts.get(t["primary_purpose"], 0) + 1
    print("\nTrials by primary purpose:")
    for p, n in sorted(purpose_counts.items(), key=lambda x: -x[1]):
        print(f"  {p:<28}{n}")
    n_treat = purpose_counts.get("TREATMENT", 0)
    print(f"Treatment-only filter would keep {n_treat} of {len(kept)}.")

    no_breast = [t for t in kept if not t["breast_mentioned"]]
    print(f"\n{len(no_breast)} trials don't mention 'breast' in title/conditions "
          "(mostly basket solid-tumor trials; skim for off-topic ones):")
    for t in no_breast:
        print(f"  {t['nct_id']}  {t['title'][:80]}")

    if TREATMENT_ONLY:
        kept = [t for t in kept if t["primary_purpose"] == "TREATMENT"]
        print(f"\nTREATMENT_ONLY is on: saving {len(kept)} trials.")

    save(kept)
    print(f"Saved {OUT_JSON} and {OUT_CSV}")

    # Quick stats for the progress report
    if kept:
        avg_len = sum(len(t["criteria_raw"]) for t in kept) / len(kept)
        print(f"Average eligibility text length: {avg_len:,.0f} characters")
        phase_counts = {}
        for t in kept:
            for p in t["phases"] or ["NA"]:
                phase_counts[p] = phase_counts.get(p, 0) + 1
        print("Trials by phase:", phase_counts)


if __name__ == "__main__":
    main()
