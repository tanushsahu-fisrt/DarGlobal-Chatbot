"""Clean the raw scraped JSON in data/raw/ and build the files the chatbot loads.

    python scripts/build_corpus.py

Outputs (in data/): listings.json, chunks.json, meta.json
"""
import glob
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "data" / "raw", ROOT / "data"

# Lines that are page furniture, not content.
JUNK = {"share", "all articles", ".", "learn more", "book now", "get in touch",
        "read more", "find your next investment"}
# If the scraped "text" contains one of these, we saved a bot-protection page by mistake.
BLOCKED = ("request unsuccessful", "incapsula incident", "access denied",
           "just a moment", "enable javascript", "captcha")
PERIOD = {"yearly": "year", "monthly": "month", "quarterly": "quarter",
          "weekly": "week", "daily": "day"}
CHUNK_WORDS, OVERLAP = 170, 30


# ---------- helpers ----------
def clean_text(text):
    lines = []
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if not ln or ln.lower() in JUNK or re.fullmatch(r"[_\-=\s.]+", ln):
            continue
        lines.append(ln)
    return "\n".join(lines)


def is_blocked(text):
    t = (text or "").lower()
    return any(b in t for b in BLOCKED)


def pub_date(text):
    """DarGlobal prints dates as dd/mm/yyyy (04/07/2022 = 4 July 2022)."""
    m = re.search(r"\b(\d{2})/(\d{2})/(\d{4})\b", (text or "")[:400])
    if not m:
        return None
    d, mo, y = map(int, m.groups())
    try:
        return datetime(y, mo, d).date().isoformat()
    except ValueError:
        return None


def num(x):
    try:
        return float(x) if x not in (None, "") else None
    except (TypeError, ValueError):
        return None


# ---------- text documents (DarGlobal pages + curated facts) ----------
def make_doc(item, stats):
    url = item.get("url", "")
    title = (item.get("title") or "").strip()
    raw = item.get("text") or ""
    if is_blocked(raw):
        stats["bot-protection page text dropped"] += 1
        raw = ""
    text = clean_text(raw)
    ctype = item.get("content_type") or (
        "press_release" if "/press/" in url else
        "blog_article" if "/blog/" in url else "company_page")
    published = pub_date(raw) if ctype == "press_release" else None

    location = item.get("location")
    if location:  # project entries often only have a title + location
        ctype = "project"
        if not text:
            stats["projects with title+location only"] += 1
        text = f"{title} is a DarGlobal development located in {location}." + \
               ("\n" + text if text else "")
    if len(text) < 80:
        stats["documents dropped (empty/too short)"] += 1
        return None
    return {"source": item.get("source", "darglobal"), "content_type": ctype,
            "title": title, "url": url, "published": published,
            "scraped_at": item.get("scraped_at"), "text": text}


def chunk(doc, start_id):
    head = f"{doc['title']} ({doc['content_type'].replace('_', ' ')}" + \
           (f", {doc['published']}" if doc.get("published") else "") + ")"
    words = doc["text"].split()
    step, out = CHUNK_WORDS - OVERLAP, []
    for i in range(0, max(len(words), 1), step):
        piece = words[i:i + CHUNK_WORDS]
        if i > 0 and len(piece) < 25:      # tail is mostly overlap
            break
        out.append({"id": start_id + len(out), "source": doc["source"],
                    "content_type": doc["content_type"], "title": doc["title"],
                    "url": doc["url"], "published": doc["published"],
                    "text": head + "\n" + " ".join(piece)})
        if i + CHUNK_WORDS >= len(words):
            break
    return out


# ---------- Wasalt listings ----------
def listing_record(item):
    purpose = str(item.get("purpose") or "").lower().strip()
    price = num(item.get("price"))
    freq, period = item.get("rent_frequency"), None
    if isinstance(freq, dict) and freq:
        period = next((k for k, v in freq.items()
                       if isinstance(v, dict) and v.get("default_freq")), next(iter(freq)))
        if not price and isinstance(freq[period], dict):
            price = num(freq[period].get("amount"))
    elif isinstance(freq, str):
        period = freq
    if purpose != "rent":
        period = None
    beds = item.get("bedrooms")
    baths = item.get("bathrooms")
    age = item.get("building_age_years")
    rec = {
        "id": item.get("id"), "property_type": item.get("property_type"),
        "category": item.get("category"), "purpose": purpose,
        "city": item.get("city"), "district": item.get("district"),
        "price": price or None, "currency": item.get("currency") or "SAR",
        "rent_period": period,
        "bedrooms": int(beds) if beds is not None else None,
        "bathrooms": int(baths) if baths is not None else None,
        "area_sqm": num(item.get("area_sqm")),
        "building_age_years": int(age) if age is not None else None,
        "amenities": [str(a.get("name") if isinstance(a, dict) else a)
                      for a in (item.get("amenities") or [])],
        "updated_at": item.get("updated_at") or "",
        "url": item.get("url"),
        # NOTE: lat/lon and any contact details are deliberately NOT kept.
    }
    rec["text"] = listing_text(rec)
    return rec


def listing_text(r):
    where = ", ".join(x for x in (r["district"], r["city"]) if x)
    t = f"{r['property_type'] or 'Property'} for {r['purpose']} in {where}."
    if r["price"]:
        t += f" Price: {r['price']:,.0f} {r['currency']}"
        if r["rent_period"]:
            t += f" per {PERIOD.get(r['rent_period'], r['rent_period'])}"
        t += "."
    else:
        t += " Price on request."
    if r["bedrooms"] is not None:
        t += f" {r['bedrooms']} bedrooms."
    if r["bathrooms"] is not None:
        t += f" {r['bathrooms']} bathrooms."
    if r["area_sqm"]:
        t += f" Area: {r['area_sqm']:g} sqm."
    if r["building_age_years"] is not None:
        t += f" Building age: {r['building_age_years']} years."
    if r["amenities"]:
        t += " Amenities: " + ", ".join(r["amenities"]) + "."
    if r["updated_at"]:
        t += f" Listing updated {r['updated_at'][:10]}."
    return t + " Source: Wasalt.sa"


# ---------- main ----------
def main():
    files = sorted(glob.glob(str(RAW / "*.json")))
    if not files:
        raise SystemExit(f"No .json files found in {RAW}. Copy your raw scraped files there.")
    stats, listings, docs = Counter(), {}, {}
    for f in files:
        data = json.loads(Path(f).read_text("utf-8"))
        if isinstance(data, dict):
            data = data.get("items") or data.get("data") or [data]
        for item in data:
            if not isinstance(item, dict):
                continue
            if "purpose" in item and "price" in item:
                rec = listing_record(item)
                key = rec["id"] or rec["url"]
                stats["duplicate listings skipped"] += key in listings
                listings[key] = rec
            elif item.get("url"):
                d = make_doc(item, stats)
                if d:
                    docs[d["url"]] = d

    chunks = []
    for d in docs.values():
        chunks += chunk(d, len(chunks))
    listings = list(listings.values())

    scraped = [d["scraped_at"] for d in docs.values() if d.get("scraped_at")]
    as_of = (max(scraped)[:10] if scraped else datetime.now(timezone.utc).date().isoformat())
    meta = {"built_at": datetime.now(timezone.utc).isoformat(), "data_as_of": as_of,
            "listings": len(listings), "documents": len(docs), "chunks": len(chunks)}
    (OUT / "listings.json").write_text(json.dumps(listings, ensure_ascii=False, indent=1), "utf-8")
    (OUT / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=1), "utf-8")
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1), "utf-8")

    print("== Build report ==")
    print(f"listings: {len(listings)} | documents: {len(docs)} | chunks: {len(chunks)} | data as of {as_of}")
    print("documents by type:", dict(Counter(d["content_type"] for d in docs.values())))
    print("listings by purpose:", dict(Counter(l["purpose"] for l in listings)))
    print("top cities:", Counter(l["city"] for l in listings).most_common(6))
    print("listings without a price:", sum(1 for l in listings if not l["price"]))
    for k, v in stats.items():
        print(f"WARNING {k}: {v}")
    if not any(d["source"] == "wasalt" for d in docs.values()):
        print("NOTE: no Wasalt company text found - add data/raw/curated_wasalt_about.json "
              "so 'What is Wasalt?' can be answered.")


if __name__ == "__main__":
    main()
