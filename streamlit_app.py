# streamlit_app.py
# Hivemind or Headless Chickens: Website + News + PDF MVP
# Sources:
#  (a) Company website crawl, user-provided URL
#  (b) News search via DuckDuckGo, fetch and extract article text
#  (c) Annual report PDF upload
#
# No hard coverage gate. Warnings only. Justification panel included.

import io
import re
import time
import html
import random
from typing import Dict, List, Tuple, Set
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import tldextract
import trafilatura

# -------------------------------
# configuration
# -------------------------------
# Stage optima updated as requested
STAGE_OPTIMUM = {"early": 5.0, "growth": 9.0, "late": 12.0}
STAGE_LABELS = ["early", "growth", "late"]

# advisory thresholds only, never block
ADVISORY_MIN_TOKENS_GOV = 300
ADVISORY_MIN_TOKENS_STRAT_OR_PART = 300

# crawl limits
MAX_PAGES = 10           # total pages to fetch from site
CRAWL_DEPTH = 1          # only same-domain links up to depth 1
REQ_TIMEOUT = 12         # seconds
UA = "Mozilla/5.0 (compatible; HivemindMVP/1.0; +https://example.com/bot)"

# news limits
NEWS_RESULTS = 6
NEWS_DAYS = 365

# semantic similarity threshold
SIM_THRESHOLD = 0.34

# lexicons
INNOVATION_TERMS = [
    "innovation","innovative","innovate","experiment","prototype","breakthrough",
    "research","r&d","invention","novel","disrupt","explore","exploration","pilot",
    "beta","hypothesis","test-and-learn","iterate","iteration","skunkworks","lab","incubate"
]
PROCESS_TERMS = [
    "efficiency","efficient","standardise","standardize","compliance","regulatory",
    "control","controls","governance","policy","policies","audit","risk","risk management",
    "brand protection","quality management","six sigma","lean","operational excellence",
    "procedures","protocol","cost discipline","process","processes"
]
ISOMORPHISM_TERMS = [
    "best practice","industry standard","benchmark","peer group","comparable",
    "regulatory requirements","compliance with","aligned with standards","iso",
    "certification","accreditation"
]
GOVERNANCE_TITLES = [
    "director","non-executive director","independent director","chair","chairman",
    "chief executive officer","ceo","chief financial officer","cfo","chief operating officer",
    "coo","chief technology officer","cto","chief risk officer","cro","executive director"
]

# prototypes for semantic sectioning
PROTOTYPES = {
    "governance": """
board composition, directors, independence, committees, audit committee,
remuneration, nomination, risk oversight, corporate governance code,
executive officers, leadership biographies, responsibilities, terms of reference
""",
    "strategy": """
strategy, strategic priorities, business model, competitive advantage,
market positioning, capital allocation, operating model, growth levers,
management discussion and analysis, md&a, ceo letter, outlook
""",
    "partnerships": """
partnerships, alliances, joint venture, ecosystem partners, customers,
suppliers, distribution agreements, channel partners, co-development,
collaboration, memorandum of understanding
"""
}

# -------------------------------
# utilities
# -------------------------------
def normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def token_count_text(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))

def ocai(pci: float, stage: str) -> Tuple[float, str]:
    cstar = STAGE_OPTIMUM[stage]
    delta = pci - cstar
    direction = "Obsolescence" if delta > 0 else "Fragmentation"
    return abs(delta), direction

def guess_company_name_from_text(text: str) -> str:
    # naive guess from title-ish phrases
    m = re.search(r"^\s*([A-Z][A-Za-z0-9&,\.\- ']{2,80})\s*$", text.split("\n")[0])
    return m.group(1).strip() if m else "COMPANY"

# -------------------------------
# robust PDF text extraction
# -------------------------------
def extract_pages_from_pdf(file_like) -> List[str]:
    # 1) PyMuPDF
    try:
        import fitz
        file_like.seek(0)
        pages = []
        with fitz.open(stream=file_like.read(), filetype="pdf") as doc:
            for p in doc:
                pages.append(normalise_whitespace(p.get_text("text")))
        if pages and any(pages):
            return pages
    except Exception:
        pass
    # 2) pdfplumber
    try:
        import pdfplumber
        file_like.seek(0)
        pages = []
        with pdfplumber.open(file_like) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                pages.append(normalise_whitespace(text))
        if pages and any(pages):
            return pages
    except Exception:
        pass
    # 3) pdfminer.six
    try:
        from pdfminer.high_level import extract_text
        file_like.seek(0)
        text = extract_text(file_like) or ""
        pages = [normalise_whitespace(t) for t in text.split("\f") if t.strip()]
        if pages and any(pages):
            return pages
    except Exception:
        pass
    return []

# -------------------------------
# website crawling and extraction
# -------------------------------
def same_domain(url: str, origin: str) -> bool:
    if not url or not origin:
        return False
    a = tldextract.extract(url)
    b = tldextract.extract(origin)
    return (a.domain, a.suffix) == (b.domain, b.suffix)

def fetch_url(url: str) -> str:
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=REQ_TIMEOUT)
        if r.status_code != 200 or "text/html" not in r.headers.get("Content-Type",""):
            return ""
        # trafilatura handles messy HTML well
        extracted = trafilatura.extract(r.text, include_comments=False, include_tables=False)
        if extracted:
            return normalise_whitespace(extracted)
        # fallback to visible text extraction
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script","style","noscript"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
        return normalise_whitespace(text)
    except Exception:
        return ""

def crawl_site(start_url: str, max_pages: int = MAX_PAGES, depth: int = CRAWL_DEPTH) -> Dict[str, str]:
    seen: Set[str] = set()
    queue: List[Tuple[str,int]] = [(start_url, 0)]
    texts: Dict[str, str] = {}
    origin = start_url

    while queue and len(texts) < max_pages:
        url, d = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        txt = fetch_url(url)
        if txt and token_count_text(txt) > 80:
            texts[url] = txt
        if d < depth:
            # collect links
            try:
                r = requests.get(url, headers={"User-Agent": UA}, timeout=REQ_TIMEOUT)
                soup = BeautifulSoup(r.text, "html.parser")
                for a in soup.find_all("a", href=True):
                    href = a["href"].strip()
                    if href.startswith("#"):
                        continue
                    next_url = urljoin(url, href)
                    if same_domain(next_url, origin) and next_url not in seen:
                        if any(ext in next_url.lower() for ext in [".pdf",".jpg",".png",".zip",".mp4",".svg"]):
                            continue
                        queue.append((next_url, d+1))
            except Exception:
                pass
    return texts  # dict[url] = extracted_text

# -------------------------------
# news search and extraction
# -------------------------------
def search_news(query: str, days: int = NEWS_DAYS, k: int = NEWS_RESULTS) -> List[Dict]:
    # DuckDuckGo News search, returns [{'title','date','source','url','body'}, ...]
    try:
        with DDGS(timeout=REQ_TIMEOUT) as ddgs:
            results = ddgs.news(keywords=query, timelimit=f"d{days}", max_results=k)
            return results or []
    except Exception:
        return []

def fetch_articles(urls: List[str]) -> Dict[str, str]:
    texts = {}
    for u in urls:
        txt = fetch_url(u)
        if txt and token_count_text(txt) > 120:
            texts[u] = txt
    return texts

# -------------------------------
# semantic sectioner
# -------------------------------
@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

class SemanticSectioner:
    def __init__(self, sim_threshold: float = SIM_THRESHOLD):
        self.enc = get_embedder()
        self.labels = list(PROTOTYPES.keys())
        self.P = self._encode_prototypes()
        self.sim_threshold = sim_threshold

    def _encode_prototypes(self):
        vecs = []
        for k in self.labels:
            v = self.enc.encode([PROTOTYPES[k].strip()], convert_to_numpy=True, normalize_embeddings=True)
            vecs.append(v[0])
        return np.vstack(vecs)

    def chunk(self, texts: List[str], min_tokens: int = 80, max_tokens: int = 260, stride: int = 160) -> List[str]:
        text = "\n\n".join(texts)
        toks = re.findall(r"\S+", text)
        chunks = []
        for i in range(0, max(1, len(toks) - min_tokens), stride):
            piece = " ".join(toks[i:i+max_tokens])
            if len(piece.split()) >= min_tokens:
                chunks.append(piece)
        if not chunks and len(text.split()) >= min_tokens:
            chunks = [text]
        return chunks

    def split(self, texts: List[str]) -> Dict[str, List[str]]:
        chunks = self.chunk(texts)
        sections = {"governance": [], "strategy": [], "partnerships": [], "other": []}
        if not chunks:
            return sections
        E = self.enc.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
        sims = cosine_similarity(E, self.P)
        best = sims.argmax(axis=1)
        maxv = sims.max(axis=1)
        for i, ch in enumerate(chunks):
            if maxv[i] >= self.sim_threshold:
                sections[self.labels[best[i]]].append(ch)
            else:
                sections["other"].append(ch)
        return sections

# -------------------------------
# features
# -------------------------------
def bag_counts(texts: List[str], vocab: List[str]) -> int:
    if not texts:
        return 0
    joined = " \n ".join(texts).lower()
    total = 0
    for term in vocab:
        total += joined.count(term.lower())
    return int(total)

def lexicon_features(sections: Dict[str, List[str]]) -> Dict[str, float]:
    texts = [t for lst in sections.values() for t in lst]
    total_tokens = sum(len(t.split()) for t in texts) or 1
    innov = bag_counts(texts, INNOVATION_TERMS)
    proc = bag_counts(texts, PROCESS_TERMS)
    iso = bag_counts(texts, ISOMORPHISM_TERMS)
    return {
        "innov_per_1k": 1000.0 * innov / total_tokens,
        "proc_per_1k": 1000.0 * proc / total_tokens,
        "iso_per_1k": 1000.0 * iso / total_tokens,
        "explore_exploit_ratio": innov / (proc + 1e-6),
    }

def homogeneity_within(sections: Dict[str, List[str]]) -> Dict[str, float]:
    texts = []
    for lst in sections.values():
        for s in lst:
            if len(s.split()) > 120:
                texts.append(s)
    if len(texts) < 3:
        return {"pairwise_mean_sim": 0.0, "pairwise_std_sim": 0.0}
    enc = get_embedder()
    E = enc.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    sims = cosine_similarity(E, E)
    tri = sims[np.triu_indices_from(sims, k=1)]
    return {"pairwise_mean_sim": float(tri.mean()), "pairwise_std_sim": float(tri.std())}

def governance_cues(sections: Dict[str, List[str]]) -> Dict[str, int]:
    gov_text = " \n ".join(sections.get("governance", []))
    lt = gov_text.lower()
    title_mentions = sum(lt.count(t) for t in GOVERNANCE_TITLES)
    names = re.findall(r"\b([A-Z][a-z]+ [A-Z][a-zA-Z\-']+)\b", gov_text)
    distinct_people = len(set(names))
    top_person_repeats = int(pd.Series(names).value_counts().iloc[0]) if names else 0
    return {
        "gov_distinct_people": int(distinct_people),
        "gov_title_mentions": int(title_mentions),
        "gov_top_person_repeats": int(top_person_repeats),
    }

def partnership_cues(sections: Dict[str, List[str]]) -> Dict[str, float]:
    text = " \n ".join(sections.get("partnerships", []) + sections.get("strategy", []))
    org_like = re.findall(r"\b([A-Z][A-Z&\-\.,]{2,})\b", text)
    blacklist = {"AND","THE","FOR","WITH","OUR","THIS","THAT"}
    orgs = [o for o in org_like if o not in blacklist]
    if not orgs:
        return {"ally_count": 0, "ally_distinct": 0, "ally_hhi": 0.0}
    counts = pd.Series(orgs).value_counts()
    shares = counts / counts.sum()
    hhi = float((shares ** 2).sum())
    return {"ally_count": int(counts.sum()), "ally_distinct": int(counts.size), "ally_hhi": hhi}

def assemble_numeric_features(sections: Dict[str, List[str]]) -> Dict[str, float]:
    feats = {}
    feats.update(lexicon_features(sections))
    feats.update(homogeneity_within(sections))
    feats.update(governance_cues(sections))
    feats.update(partnership_cues(sections))
    return feats

# -------------------------------
# transparent scoring
# -------------------------------
def heuristic_pci_and_stage(numeric: Dict[str, float]) -> Tuple[float, str]:
    pci = 10.0 \
        + 2.5 * np.tanh(numeric.get("proc_per_1k", 0.0) / 10.0) \
        + 2.0 * np.tanh(numeric.get("pairwise_mean_sim", 0.0) * 2.0) \
        + 1.0 * np.tanh(numeric.get("ally_hhi", 0.0) * 3.0) \
        - 2.5 * np.tanh(numeric.get("innov_per_1k", 0.0) / 10.0)
    pci = float(np.clip(pci, 4.0, 20.0))

    stage_logits = np.array([
        - numeric.get("proc_per_1k", 0.0) - 0.1 * numeric.get("gov_distinct_people", 0.0),
        0.2 * numeric.get("proc_per_1k", 0.0),
        0.5 * numeric.get("proc_per_1k", 0.0) + 0.1 * numeric.get("gov_distinct_people", 0.0)
    ])
    stage = STAGE_LABELS[int(stage_logits.argmax())]
    return pci, stage

# -------------------------------
# justification helpers
# -------------------------------
def top_similar_chunks(all_texts: List[str], sectioner: SemanticSectioner, k_per_bucket=2):
    chunks = sectioner.chunk(all_texts)
    if not chunks:
        return {"governance": [], "strategy": [], "partnerships": []}
    E = sectioner.enc.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    sims = cosine_similarity(E, sectioner.P)
    labels = sectioner.labels
    out = {}
    for j, label in enumerate(labels):
        idx = np.argsort(-sims[:, j])[:k_per_bucket]
        out[label] = [{"score": float(sims[i, j]), "text": chunks[i][:800]} for i in idx if sims[i, j] > 0]
    return out

def keyword_contributions(sections):
    joined = {
        "all": " \n ".join([t for lst in sections.values() for t in lst]).lower(),
    }
    def count_terms(text, terms):
        counts = [(term, text.count(term.lower())) for term in terms]
        counts = [(t, c) for t, c in counts if c > 0]
        return sorted(counts, key=lambda x: -x[1])[:10]
    return {
        "innovation_terms": count_terms(joined["all"], INNOVATION_TERMS),
        "process_terms": count_terms(joined["all"], PROCESS_TERMS),
        "isomorphism_terms": count_terms(joined["all"], ISOMORPHISM_TERMS)
    }

def extract_named_entities_light(sections):
    gov_text = " \n ".join(sections.get("governance", []))
    names = re.findall(r"\b([A-Z][a-z]+ [A-Z][a-zA-Z\-']+)\b", gov_text)
    names_top = list(pd.Series(names).value_counts().head(8).items()) if names else []
    part_text = " \n ".join(sections.get("partnerships", []) + sections.get("strategy", []))
    org_like = re.findall(r"\b([A-Z][A-Z&\-\.,]{2,})\b", part_text)
    blacklist = {"AND","THE","FOR","WITH","OUR","THIS","THAT"}
    orgs = [o for o in org_like if o not in blacklist]
    orgs_top = list(pd.Series(orgs).value_counts().head(10).items()) if orgs else []
    return {"people_top": names_top, "orgs_top": orgs_top}

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Hivemind or Headless Chickens", layout="centered")
st.title("Hivemind or Headless Chickens")

st.caption("Paste a company website, optionally enable news search, and, if you like, upload an annual report PDF. The app analyses all sources together.")

with st.form("inputs"):
    website_url = st.text_input("Company website URL, include https://", value="", placeholder="https://example.com")
    news_toggle = st.checkbox("Fetch recent news about the company")
    news_query = ""
    if news_toggle:
        news_query = st.text_input("News search query", value="", placeholder="Company name, e.g., Acme Corp")
    pdf_file = st.file_uploader("Optional, upload annual report (PDF)", type=["pdf"])
    submitted = st.form_submit_button("Analyse")

if not submitted:
    st.stop()

all_texts: List[str] = []
sources_summary = []

# 1) Website crawl
if website_url.strip():
    st.info("Crawling website, within domain, shallow depth")
    t0 = time.time()
    site_texts = crawl_site(website_url.strip(), max_pages=MAX_PAGES, depth=CRAWL_DEPTH)
    for u, txt in site_texts.items():
        all_texts.append(txt)
    st.success(f"Fetched {len(site_texts)} pages in {time.time()-t0:.1f}s")
    if site_texts:
        sources_summary.append(("website_pages", len(site_texts)))

# 2) News fetch
news_results = []
if news_toggle and news_query.strip():
    st.info("Searching and fetching news")
    results = search_news(news_query.strip(), days=NEWS_DAYS, k=NEWS_RESULTS)
    urls = [r.get("url") for r in results if r.get("url")]
    articles = fetch_articles(urls)
    for u, txt in articles.items():
        all_texts.append(txt)
    news_results = results
    st.success(f"Pulled {len(articles)} articles")
    if articles:
        sources_summary.append(("news_articles", len(articles)))

# 3) PDF upload
if pdf_file is not None:
    st.info("Parsing PDF")
    pages = extract_pages_from_pdf(io.BytesIO(pdf_file.read()))
    if pages:
        all_texts.extend(pages)
        sources_summary.append(("pdf_pages", len(pages)))
        st.success(f"Parsed {len(pages)} PDF pages")
    else:
        st.warning("Could not parse text from the PDF")

# If nothing collected, stop
if not all_texts:
    st.error("No text collected from website, news, or PDF. Please provide at least one source.")
    st.stop()

# Company name heuristic
company_hint = "COMPANY"
if website_url:
    parsed = urlparse(website_url)
    company_hint = parsed.hostname.split(".")[0].upper()
elif news_query:
    company_hint = news_query.strip()

st.write("**Sources used**:", ", ".join([f"{k}={v}" for k, v in sources_summary]) or "none")

# Semantic sectioning
st.info("Finding governance, strategy, and partnerships content")
sectioner = SemanticSectioner(sim_threshold=SIM_THRESHOLD)
sections = sectioner.split(all_texts)

# Advisory warnings only
gov_tokens = sum(len(s.split()) for s in sections.get("governance", []))
strat_tokens = sum(len(s.split()) for s in sections.get("strategy", []))
part_tokens = sum(len(s.split()) for s in sections.get("partnerships", []))
if gov_tokens < ADVISORY_MIN_TOKENS_GOV:
    st.warning(f"Governance like content appears limited, {gov_tokens} tokens.")
if (strat_tokens + part_tokens) < ADVISORY_MIN_TOKENS_STRAT_OR_PART:
    st.warning(f"Strategy or partnerships like content appears limited, {strat_tokens + part_tokens} tokens.")

# Features and scoring
st.info("Extracting features and scoring")
numeric = assemble_numeric_features(sections)
pci, stage = heuristic_pci_and_stage(numeric)
ocai_val, direction = ocai(pci, stage)

c1, c2, c3 = st.columns(3)
c1.metric("PCI", f"{pci:.1f}")
c2.metric("Stage", stage.capitalize())
c3.metric("OCAI", f"{ocai_val:.1f}", direction)

# Optional, show a small news table
if news_results:
    st.subheader("News used")
    news_table = []
    for r in news_results:
        news_table.append({
            "title": r.get("title",""),
            "date": str(r.get("date","")),
            "source": r.get("source",""),
            "url": r.get("url","")
        })
    st.dataframe(pd.DataFrame(news_table))

# Justifications
st.subheader("Why the model thinks this")
with st.container(border=True):
    reps = top_similar_chunks(all_texts, sectioner, k_per_bucket=2)
    st.markdown("**Representative passages by category**")
    for label in ["governance", "strategy", "partnerships"]:
        if reps.get(label):
            st.markdown(f"*{label.capitalize()}*")
            for item in reps[label]:
                st.caption(f"Similarity {item['score']:.2f}")
                st.write("> " + item["text"])

    st.markdown("**Keyword signals**")
    keys = keyword_contributions(sections)
    col_a, col_b, col_c = st.columns(3)
    col_a.write("Innovation")
    if keys["innovation_terms"]:
        col_a.table(pd.DataFrame(keys["innovation_terms"], columns=["term","count"]))
    else:
        col_a.write("No salient terms")

    col_b.write("Process")
    if keys["process_terms"]:
        col_b.table(pd.DataFrame(keys["process_terms"], columns=["term","count"]))
    else:
        col_b.write("No salient terms")

    col_c.write("Isomorphism")
    if keys["isomorphism_terms"]:
        col_c.table(pd.DataFrame(keys["isomorphism_terms"], columns=["term","count"]))
    else:
        col_c.write("No salient terms")

    st.markdown("**Entities mentioned**")
    ents = extract_named_entities_light(sections)
    c1, c2 = st.columns(2)
    c1.write("People, governance context")
    if ents["people_top"]:
        c1.table(pd.DataFrame(ents["people_top"], columns=["name","mentions"]))
    else:
        c1.write("None detected")
    c2.write("Organisations, partnership context")
    if ents["orgs_top"]:
        c2.table(pd.DataFrame(ents["orgs_top"], columns=["organisation","mentions"]))
    else:
        c2.write("None detected")

st.caption("Free MVP. Website crawl is shallow by design, news via DuckDuckGo, and optional PDF parsing.")
