"""Retrieval layer.

Two kinds of data need two kinds of search:
  * Wasalt listings  -> structured filters (city, type, price, bedrooms...) parsed from the question
  * DarGlobal pages  -> BM25 keyword search over text chunks
The LLM only sees what this module retrieves.
"""
import json
import re
import statistics
from pathlib import Path

from rank_bm25 import BM25Okapi

DATA = Path(__file__).resolve().parent.parent / "data"

_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("""a an the of in on at for to and or is are was were be do does did what which who how
when where why me my i you your we our can could would should show list tell give find about with
from any there their it its this that these those please""".split())
EXPAND = {"ceo": "chief executive officer founder", "founder": "chief executive officer founder",
          "founded": "founder chief executive officer", "founding": "founder chief executive officer",
          "boss": "chief executive officer", "listed": "london stock exchange",
          "headquarters": "offices global presence", "hq": "offices global presence"}
PER_YEAR = {"yearly": 1, "monthly": 12, "quarterly": 4, "weekly": 52, "daily": 365}
LABEL = {"yearly": "year", "monthly": "month", "quarterly": "quarter", "weekly": "week", "daily": "day"}

DAR_HINTS = re.compile(r"dar\s?global|dar al arkan|trump|sidra|missoni|lamborghini|aston martin|pagani|"
                       r"versace|elie saab|cavalli|w residences|da ?vinci|urban oasis|press release|"
                       r"investor|\bceo\b|founder|london stock|\bblog\b", re.I)
LISTING_WORDS = re.compile(r"\b(listings?|propert(?:y|ies)|apartments?|flats?|villas?|lands?|plots?|rent|"
                           r"rental|buy|for sale|bedrooms?|cheapest|prices?)\b", re.I)
GREETING = re.compile(r"\s*(hi+|hello|hey|salam|assalam[a-z ]*|thanks|thank you|ok|okay|"
                      r"good (morning|evening))[\s!.?]*", re.I)

GREETING_REPLY = ("Hi! I can answer questions about DarGlobal (projects, press releases, company "
                  "information) and about Wasalt property listings. Try: \"Who founded DarGlobal?\" "
                  "or \"3-bedroom apartments for rent under 50,000 SAR\".")
SCOPE_REPLY = ("I couldn't find anything about that in my data. I only cover DarGlobal (projects, press, "
               "company information) and Wasalt property listings. Try rephrasing, or ask about a "
               "specific project, city, property type or budget.")


def _stem(w):
    """Very light stemming so 'founded'/'founder', 'residences'/'residence' can match."""
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def tokens(text, expand=False):
    t = re.sub(r"dar\s+global", "darglobal", text.lower())
    words = [w for w in _WORD.findall(t) if w not in _STOP]
    if expand:
        for w in list(words):
            words += EXPAND.get(w, "").split()
    return [_stem(w) for w in words]


_NUM = r"(\d[\d,]*(?:\.\d+)?)\s*(k|m|mn|million|thousand)?(?![a-z0-9])"
_MAX = re.compile(r"(?:under|below|less than|up to|upto|max(?:imum)?(?: price)?(?: of)?|within|at most|"
                  r"budget(?: of| is)?|cheaper than)\s*(?:sar|sr|riyals?)?\s*" + _NUM)
_MIN = re.compile(r"(?:over|above|more than|at least|min(?:imum)?(?: price)?(?: of)?|starting (?:from|at))"
                  r"\s*(?:sar|sr|riyals?)?\s*" + _NUM)
_BETWEEN = re.compile(r"between\s*(?:sar\s*)?" + _NUM + r"\s*(?:and|to|-)\s*(?:sar\s*)?" + _NUM)
_AREA = re.compile(r"(under|below|less than|up to|max(?:imum)?|over|above|more than|at least|min(?:imum)?)"
                   r"\s*(\d[\d,]*(?:\.\d+)?)\s*(?:sqm|sq\.?\s?m|m2|m²|square met)")
_UNIT_TAIL = re.compile(r"\s*(bed|br\b|bhk|bath|sq|m2|m²|square|year|yr|month|floor|km)")


class Knowledge:
    def __init__(self):
        def load(name, default):
            p = DATA / name
            return json.loads(p.read_text("utf-8")) if p.exists() else default

        self.listings = load("listings.json", [])
        self.chunks = load("chunks.json", [])
        self.meta = load("meta.json", {})
        self.as_of = self.meta.get("data_as_of", "unknown date")
        self.bm25 = BM25Okapi([tokens(c["text"]) for c in self.chunks]) if self.chunks else None
        self.cities = self._names("city")
        self.districts = self._names("district")
        self.types = sorted({(l.get("property_type") or "").lower()
                             for l in self.listings if l.get("property_type")}, key=len, reverse=True)

    def _names(self, field):
        return {l[field].lower(): l[field] for l in self.listings
                if l.get(field) and len(l[field]) >= 4}

    # ---------- understanding the question ----------
    @staticmethod
    def _value(s, m, ni=1, si=2):
        num = float(m.group(ni).replace(",", ""))
        suf = m.group(si) or ""
        num *= {"k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "million": 1e6}.get(suf, 1)
        if _UNIT_TAIL.match(s[m.end():m.end() + 14]):
            return None            # "3 bedrooms", "200 sqm" are not prices
        if num < 1000 and not suf:
            return None
        return num

    def parse_filters(self, q):
        s = " " + q.lower().replace("’", "'") + " "
        f = {}
        if re.search(r"\b(rent|rental|rentals|renting|lease|leasing)\b", s):
            f["purpose"] = "rent"
        elif re.search(r"\b(buy|buying|sale|sell|selling|purchase)\b", s):
            f["purpose"] = "sale"
        for t in self.types:
            if re.search(r"\b" + re.escape(t) + r"(?:s|es)?\b", s):
                f["type"] = t
                break
        else:
            if "apartment" in self.types and re.search(r"\bflats?\b", s):
                f["type"] = "apartment"
            elif "land" in self.types and re.search(r"\bplots?\b", s):
                f["type"] = "land"
        for key, names in (("cities", self.cities), ("districts", self.districts)):
            hits = [names[n] for n in sorted(names, key=len, reverse=True)
                    if re.search(r"\b" + re.escape(n) + r"\b", s)]
            if hits:
                f[key] = hits
        if re.search(r"\b(monthly|per month|a month|/month)\b", s):
            f["monthly"] = True
        for m in _AREA.finditer(s):
            v = float(m.group(2).replace(",", ""))
            low = m.group(1).startswith(("under", "below", "less", "up", "max"))
            f["max_area" if low else "min_area"] = v
        m = _BETWEEN.search(s)
        if m:
            lo, hi = self._value(s, m, 1, 2), self._value(s, m, 3, 4)
            if lo and hi:
                f["min_price"], f["max_price"] = min(lo, hi), max(lo, hi)
        for key, rx in (("max_price", _MAX), ("min_price", _MIN)):
            if key not in f:
                for m in rx.finditer(s):
                    v = self._value(s, m)
                    if v:
                        f[key] = v
                        break
        m = re.search(r"(\d+)\s*[- ]?\s*(\+)?\s*(?:bed(?:room)?s?|br|bhk)\b", s)
        if m:
            pre = s[max(0, m.start() - 14):m.start()]
            if m.group(2) or re.search(r"(at least|min|minimum|more than|over|above)\s*$", pre):
                f["min_beds"] = int(m.group(1))
            else:
                f["beds"] = int(m.group(1))
        if re.search(r"\b(cheapest|lowest|cheap|affordable|least expensive)\b", s):
            f["sort"] = "price_asc"
        elif re.search(r"\b(most expensive|highest|priciest)\b", s):
            f["sort"] = "price_desc"
        return f

    # ---------- listings ----------
    def filter_listings(self, f):
        out = []
        for l in self.listings:
            if f.get("purpose") and l["purpose"] != f["purpose"]:
                continue
            if f.get("type") and (l.get("property_type") or "").lower() != f["type"]:
                continue
            if f.get("cities") and l.get("city") not in f["cities"]:
                continue
            if f.get("districts") and l.get("district") not in f["districts"]:
                continue
            if "max_price" in f or "min_price" in f:
                p = l.get("price")
                if not p:
                    continue
                if f.get("monthly") and l["purpose"] == "rent":
                    p = p * PER_YEAR.get(l.get("rent_period"), 1) / 12   # -> per month
                if p > f.get("max_price", float("inf")) or p < f.get("min_price", 0):
                    continue
            b = l.get("bedrooms")
            if "beds" in f and b != f["beds"]:
                continue
            if "min_beds" in f and (b is None or b < f["min_beds"]):
                continue
            a = l.get("area_sqm")
            if "max_area" in f and (not a or a > f["max_area"]):
                continue
            if "min_area" in f and (not a or a < f["min_area"]):
                continue
            out.append(l)
        sort = f.get("sort")
        if sort == "price_asc":
            out.sort(key=lambda l: l["price"] or float("inf"))
        elif sort == "price_desc":
            out.sort(key=lambda l: -(l["price"] or 0))
        else:
            out.sort(key=lambda l: l.get("updated_at") or "", reverse=True)
        return out

    def relax_hints(self, f):
        hints = []
        for key in ("max_price", "min_price", "beds", "min_beds", "max_area", "min_area",
                    "districts", "cities", "type"):
            if key in f:
                n = len(self.filter_listings({k: v for k, v in f.items() if k != key}))
                if n:
                    hints.append((n, key))
        names = {"max_price": "the maximum price", "min_price": "the minimum price", "beds": "the bedroom count",
                 "min_beds": "the bedroom minimum", "max_area": "the area limit", "min_area": "the area minimum",
                 "districts": "the district", "cities": "the city", "type": "the property type"}
        return [f"without {names[k]} there are {n} listings" for n, k in sorted(hints, reverse=True)[:3]]

    @staticmethod
    def describe(f):
        bits = []
        if "purpose" in f: bits.append(f"for {f['purpose']}")
        if "type" in f: bits.append(f["type"])
        if "cities" in f: bits.append("city " + "/".join(f["cities"]))
        if "districts" in f: bits.append("district " + "/".join(f["districts"]))
        if "beds" in f: bits.append(f"{f['beds']} bedrooms")
        if "min_beds" in f: bits.append(f"at least {f['min_beds']} bedrooms")
        if "min_price" in f: bits.append(f"min price {f['min_price']:,.0f} SAR" + (" per month" if f.get("monthly") else ""))
        if "max_price" in f: bits.append(f"max price {f['max_price']:,.0f} SAR" + (" per month" if f.get("monthly") else ""))
        if "min_area" in f: bits.append(f"min area {f['min_area']:g} sqm")
        if "max_area" in f: bits.append(f"max area {f['max_area']:g} sqm")
        return ", ".join(bits) or "none (all listings)"

    @staticmethod
    def fmt_listing(l, label):
        if l.get("price"):
            price = f"SAR {l['price']:,.0f}"
            if l["purpose"] == "rent" and l.get("rent_period"):
                price += "/" + LABEL.get(l["rent_period"], l["rent_period"])
        else:
            price = "price on request"
        bits = [f"{l.get('property_type') or 'Property'} for {l['purpose']}",
                ", ".join(x for x in (l.get("district"), l.get("city")) if x), price]
        if l.get("bedrooms") is not None:
            bits.append(f"{l['bedrooms']} bedrooms")
        if l.get("area_sqm"):
            bits.append(f"{l['area_sqm']:g} sqm")
        if l.get("building_age_years") is not None:
            bits.append(f"{l['building_age_years']} yrs old")
        return f"[{label}] " + " | ".join(bits) + f" | updated {(l.get('updated_at') or '')[:10]} | {l.get('url')}"

    # ---------- text documents ----------
    def search_text(self, q, k=4):
        if not self.bm25:
            return []
        qt = tokens(q, expand=True)
        if not qt:
            return []
        scores = self.bm25.get_scores(qt)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k * 3]
        if not order or scores[order[0]] <= 0:
            return []
        top, picked, per_doc = scores[order[0]], [], {}
        for i in order:
            c = self.chunks[i]
            if scores[i] < 0.4 * top or per_doc.get(c["url"], 0) >= 2:
                continue
            per_doc[c["url"]] = per_doc.get(c["url"], 0) + 1
            picked.append(c)
            if len(picked) == k:
                break
        return picked

    # ---------- main entry ----------
    def retrieve(self, question, prev_user=(), n_listings=6):
        q = question.strip()
        if GREETING.fullmatch(q):
            return self._direct(GREETING_REPLY)

        f = self.parse_filters(q)
        short = len(q.split()) <= 8
        if short and prev_user and (set(f) or LISTING_WORDS.search(q)):   # follow-up question
            inherited = {}
            for m in prev_user:
                inherited.update(self.parse_filters(m))
            inherited.pop("sort", None)
            f = {**inherited, **f}

        dar = bool(DAR_HINTS.search(q))
        wasalt = bool(re.search(r"\bwasalt\b", q, re.I))
        filt = set(f) - {"sort", "monthly"}
        run_listings = bool(filt or LISTING_WORDS.search(q)) and not (dar and not wasalt)
        run_text = (not run_listings) or dar or wasalt
        text_q = q
        if len(q.split()) <= 6 and prev_user and not run_listings:
            text_q = prev_user[-1] + " " + q

        parts, sources, lines, snips = [], [], [], []
        matches = []
        if run_listings:
            matches = self.filter_listings(f)
            shown = matches[:n_listings]
            parts.append(f"WASALT LISTINGS (snapshot as of {self.as_of}). Filters understood: {self.describe(f)}. "
                         f"{len(matches)} matching listings out of {len(self.listings)} total; showing {len(shown)}.")
            for i, l in enumerate(shown, 1):
                line = self.fmt_listing(l, f"L{i}")
                lines.append(line)
                parts.append(line)
                sources.append({"label": f"L{i}", "source": "wasalt", "url": l.get("url"),
                                "title": f"{l.get('property_type')} for {l['purpose']} - "
                                         f"{l.get('district') or ''}, {l.get('city') or ''}"})
            prices = [l["price"] for l in matches if l.get("price")]
            if len(prices) >= 3:
                parts.append(f"Price stats over all {len(prices)} matches: min {min(prices):,.0f}, "
                             f"median {statistics.median(prices):,.0f}, max {max(prices):,.0f} SAR.")

        hits = self.search_text(text_q) if run_text else []
        if hits:
            parts.append(f"DOCUMENTS (public pages scraped {self.as_of}):")
            seen = {}
            for c in hits:
                if c["url"] not in seen:
                    seen[c["url"]] = f"D{len(seen) + 1}"
                    sources.append({"label": seen[c["url"]], "source": c["source"], "url": c["url"],
                                    "title": c["title"]})
                lab = seen[c["url"]]
                date = f", {c['published']}" if c.get("published") else ""
                parts.append(f"[{lab}] {c['title']} ({c['content_type']}{date}) - {c['url']}\n{c['text'][:1100]}")
                snips.append((c["title"], c["url"], c["text"][:300]))

        if run_listings and not matches and not hits:
            msg = "I couldn't find Wasalt listings that match: " + self.describe(f) + "."
            hints = self.relax_hints(f)
            if hints:
                msg += " Try loosening a filter - " + "; ".join(hints) + "."
            return self._direct(msg + f"\n\nSource: Wasalt.sa (snapshot as of {self.as_of}).", f)
        if not parts:
            return self._direct(SCOPE_REPLY)

        return {"direct": None, "context": "\n\n".join(parts), "sources": sources, "filters": f,
                "used_listings": run_listings and bool(matches), "lines": lines, "snips": snips}

    @staticmethod
    def _direct(text, f=None):
        return {"direct": text, "context": "", "sources": [], "filters": f or {},
                "used_listings": False, "lines": [], "snips": []}

    def plain_answer(self, r):
        """Used when the LLM is unavailable: show the retrieved data directly."""
        out = ["The AI model is busy or has reached its free limit, so here are the closest matches "
               "from my data:"]
        out += ["- " + l for l in r["lines"]]
        out += [f"- {t}: {s[:200]}... ({u})" for t, u, s in r["snips"][:3]]
        if r["used_listings"]:
            out.append("\nSource: Wasalt.sa")
        return "\n".join(out)
