import io
import os
import re
import tempfile
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pdfplumber
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# -------------------------------
# configurable bits
# -------------------------------
STAGE_OPTIMUM = {"early": 7.0, "growth": 11.0, "late": 15.0}
STAGE_LABELS = ["early", "growth", "late"]

MIN_WORDS = 3000
MIN_SECTIONS_REQUIRED = {
    "governance": 1,
    "strategy_or_partnership": 1,
}

SECTION_KEYWORDS = {
    "governance": ["corporate governance", "board of directors", "directors", "board", "leadership", "management"],
    "strategy": ["strategy", "strategic", "business model", "letter to shareholders", "ceo letter", "md&a", "management discussion", "operations review"],
    "partnerships": ["partnerships", "alliances", "joint venture", "ecosystem", "customers", "suppliers", "distributors"],
}

# seed lexicons, extend later per sector
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

# -------------------------------
# helpers
# -------------------------------
def normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def extract_pages_from_pdf(file_like):
    pages = []
    with pdfplumber.open(file_like) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text.strip())
    return pages


HEADER_RE = re.compile(r"^\s*(\d{0,2}\.?\s*)?([A-Z][A-Za-z&/ \-]{3,60})\s*$")

def split_into_sections(pages: List[str]) -> Dict[str, List[str]]:
    sections = {"governance": [], "strategy": [], "partnerships": [], "other": []}
    current = "other"
    for raw in pages:
        lines = raw.split("\n")
        buf = []
        for line in lines:
            if HEADER_RE.match(line.strip()):
                if buf:
                    sections[current].append("\n".join(buf).strip())
                    buf = []
                lowered = line.lower()
                if any(k in lowered for k in SECTION_KEYWORDS["governance"]):
                    current = "governance"
                elif any(k in lowered for k in SECTION_KEYWORDS["strategy"]):
                    current = "strategy"
                elif any(k in lowered for k in SECTION_KEYWORDS["partnerships"]):
                    current = "partnerships"
                else:
                    current = "other"
                buf.append(line)
            else:
                buf.append(line)
        if buf:
            sections[current].append("\n".join(buf).strip())
    return sections

def section_coverage(sections: Dict[str, List[str]]) -> Dict[str, int]:
    return {k: sum(1 for s in v if len(s.split()) > 100) for k, v in sections.items()}

def coverage_gate(pages: List[str], sections: Dict[str, List[str]]) -> Tuple[bool, List[str]]:
    total_words = sum(len(p.split()) for p in pages)
    cov = section_coverage(sections)
    has_gov = cov.get("governance", 0) >= MIN_SECTIONS_REQUIRED["governance"]
    has_strat_or_part = (cov.get("strategy", 0) + cov.get("partnerships", 0)) >= MIN_SECTIONS_REQUIRED["strategy_or_partnership"]
    ok = (total_words >= MIN_WORDS) and has_gov and has_strat_or_part
    missing = []
    if total_words < MIN_WORDS:
        missing.append(f"minimum words {MIN_WORDS}")
    if not has_gov:
        missing.append("governance section")
    if not has_strat_or_part:
        missing.append("strategy or partnerships section")
    return ok, missing

def guess_company_name(pages: List[str]) -> str:
    if not pages:
        return "COMPANY"
    first = pages[0]
    m = re.search(r"([A-Z][A-Za-z&,\.\- ]{2,80})\b(annual report|annual|report)", first, re.I)
    if m:
        return m.group(1).strip()
    return "COMPANY"

def bag_counts(texts: List[str], vocab: List[str]) -> int:
    if not texts:
        return 0
    v = CountVectorizer(vocabulary=[t.lower() for t in vocab], lowercase=True, token_pattern=r"(?u)\b[\w\-&]+\b")
    X = v.fit_transform(texts)
    return int(X.sum())

def lexicon_features(sections: Dict[str, List[str]]) -> Dict[str, float]:
    texts = [t for lst in sections.values() for t in lst]
    innov = bag_counts(texts, INNOVATION_TERMS)
    proc = bag_counts(texts, PROCESS_TERMS)
    iso = bag_counts(texts, ISOMORPHISM_TERMS)
    total_tokens = sum(len(t.split()) for t in texts) or 1
    feats = {
        "innov_per_1k": 1000.0 * innov / total_tokens,
        "proc_per_1k": 1000.0 * proc / total_tokens,
        "iso_per_1k": 1000.0 * iso / total_tokens,
        "explore_exploit_ratio": innov / (proc + 1e-6),
    }
    return feats

@st.cache_resource(show_spinner=False)
def get_embedder():
    # small, fast, free
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

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
    # regex counts of titles and unique name-like tokens as a lightweight proxy
    gov_text = " \n ".join(sections.get("governance", []))
    title_mentions = 0
    lt = gov_text.lower()
    for t in GOVERNANCE_TITLES:
        title_mentions += lt.count(t)
    # naive name proxy, capitalised words pairs
    names = re.findall(r"\b([A-Z][a-z]+ [A-Z][a-zA-Z\-']+)\b", gov_text)
    distinct_people = len(set(names))
    top_person_repeats = 0
    if names:
        counts = pd.Series(names).value_counts()
        top_person_repeats = int(counts.iloc[0])
    return {
        "gov_distinct_people": int(distinct_people),
        "gov_title_mentions": int(title_mentions),
        "gov_top_person_repeats": int(top_person_repeats),
    }

def partnership_cues(sections: Dict[str, List[str]]) -> Dict[str, float]:
    text = " \n ".join(sections.get("partnerships", []) + sections.get("strategy", []))
    # crude org proxy, all caps tokens of length >= 3, minus common words
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

def heuristic_pci_and_stage(numeric: Dict[str, float]) -> Tuple[float, str]:
    # transparent, rule based baseline
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

def ocai(pci: float, stage: str) -> Tuple[float, str]:
    cstar = STAGE_OPTIMUM[stage]
    delta = pci - cstar
    direction = "Obsolescence" if delta > 0 else "Fragmentation"
    return abs(delta), direction

# -------------------------------
# streamlit UI
# -------------------------------
st.set_page_config(page_title="Hivemind or Headless Chickens", layout="centered")
st.title("Hivemind or Headless Chickens")
st.caption("Upload a single annual report PDF. The app estimates conformity and stage adjusted risk from the PDF alone. If key sections are missing, it will say it cannot provide an answer.")

uploaded = st.file_uploader("Upload annual report (PDF)", type=["pdf"])

with st.expander("Settings", expanded=False):
    st.write("Default thresholds are conservative to avoid overconfident outputs.")
    st.write(f"Minimum words: {MIN_WORDS}")
    st.write("Required sections: Governance, and either Strategy or Partnerships")

if uploaded is None:
    st.info("Please upload a PDF to begin.")
    st.stop()

# keep file in memory for PyMuPDF
pdf_bytes = io.BytesIO(uploaded.read())

st.info("Parsing PDF")
pages = extract_pages_from_pdf(pdf_bytes)
sections = split_into_sections(pages)
ok, missing = coverage_gate(pages, sections)

if not ok:
    st.error("Cannot provide. Missing: " + ", ".join(missing))
    with st.expander("Section coverage details"):
        st.json(section_coverage(sections))
    st.stop()

company = guess_company_name(pages)
st.success(f"Detected company: {company}")

st.info("Extracting features")
numeric = assemble_numeric_features(sections)

st.info("Scoring")
pci, stage = heuristic_pci_and_stage(numeric)
ocai_val, direction = ocai(pci, stage)

c1, c2, c3 = st.columns(3)
c1.metric("PCI", f"{pci:.1f}")
c2.metric("Stage", stage.capitalize())
c3.metric("OCAI", f"{ocai_val:.1f}", direction)

with st.expander("Evidence and features"):
    st.write("Key numeric features")
    st.json(numeric)

with st.expander("Section coverage"):
    st.json(section_coverage(sections))

st.caption("This is a free MVP that uses only the uploaded PDF. It does not enrich with external data. Extend lexicons and heuristics for sector specific performance.")
