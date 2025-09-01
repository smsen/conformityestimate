# streamlit_app.py
# Hivemind or Headless Chickens, free MVP with semantic sectioning
# Gate removed: the app never rejects, it only warns when content looks thin.

import io
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# -------------------------------
# configuration
# -------------------------------
STAGE_OPTIMUM = {"early": 5.0, "growth": 9.0, "late": 12.0}
STAGE_LABELS = ["early", "growth", "late"]

# thresholds are now advisory only, they do not block scoring
ADVISORY_MIN_WORDS_TOTAL = 3000
ADVISORY_MIN_TOKENS_GOV = 300
ADVISORY_MIN_TOKENS_STRAT_OR_PART = 300

# seed lexicons
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

def ocai(pci: float, stage: str) -> Tuple[float, str]:
    cstar = STAGE_OPTIMUM[stage]
    delta = pci - cstar
    direction = "Obsolescence" if delta > 0 else "Fragmentation"
    return abs(delta), direction

def token_count(lst: List[str]) -> int:
    return sum(len(s.split()) for s in lst)

# -------------------------------
# robust PDF text extraction
# -------------------------------
def extract_pages_from_pdf(file_like) -> List[str]:
    """Try PyMuPDF, then pdfplumber, then pdfminer.six, otherwise raise a clear error."""
    # 1) PyMuPDF
    try:
        import fitz  # installed via 'pymupdf'
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
    raise RuntimeError("Cannot parse PDF with available parsers")

def guess_company_name(pages: List[str]) -> str:
    if not pages:
        return "COMPANY"
    first = pages[0]
    m = re.search(r"([A-Z][A-Za-z&,\.\- ]{2,80})\b(annual report|annual|report)", first, re.I)
    return m.group(1).strip() if m else "COMPANY"

# -------------------------------
# semantic sectioner
# -------------------------------
@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

class SemanticSectioner:
    def __init__(self, sim_threshold: float = 0.34):
        self.enc = get_embedder()
        self.labels = list(PROTOTYPES.keys())
        self.P = self._encode_prototypes()
        self.sim_threshold = sim_threshold

    def _encode_prototypes(self):
        vecs = []
        for k in self.labels:
            v = self.enc.encode([PROTOTYPES[k].strip()], convert_to_numpy=True, normalize_embeddings=True)
            vecs.append(v[0])
        return np.vstack(vecs)  # [3, d]

    def _chunk(self, pages: List[str], min_tokens: int = 80, max_tokens: int = 260, stride: int = 160) -> List[str]:
        text = "\n\n".join(pages)
        toks = re.findall(r"\S+", text)
        chunks = []
        for i in range(0, max(1, len(toks) - min_tokens), stride):
            piece = " ".join(toks[i:i+max_tokens])
            if len(piece.split()) >= min_tokens:
                chunks.append(piece)
        if not chunks and len(text.split()) >= min_tokens:
            chunks = [text]
        return chunks

    def split(self, pages: List[str]) -> Dict[str, List[str]]:
        chunks = self._chunk(pages)
        sections = {"governance": [], "strategy": [], "partnerships": [], "other": []}
        if not chunks:
            return sections
        E = self.enc.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
        sims = cosine_similarity(E, self.P)  # [n, 3]
        best = sims.argmax(axis=1)
        maxv = sims.max(axis=1)
        for i, ch in enumerate(chunks):
            if maxv[i] >= self.sim_threshold:
                sections[self.labels[best[i]]].append(ch)
            else:
                sections["other"].append(ch)
        return sections

# -------------------------------
# features from semantic sections
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
# streamlit UI
# -------------------------------
st.set_page_config(page_title="Hivemind or Headless Chickens", layout="centered")
st.title("Hivemind or Headless Chickens")
st.caption("Upload one annual report PDF. The app uses semantic sectioning to estimate conformity and stage adjusted risk from the PDF alone. It never rejects, it only warns when content is thin.")

uploaded = st.file_uploader("Upload annual report (PDF)", type=["pdf"])

with st.expander("Advisory settings", expanded=False):
    st.write("These thresholds only trigger warnings and do not block scoring.")
    st.write(f"Advisory minimum total words: {ADVISORY_MIN_WORDS_TOTAL}")
    st.write(f"Advisory minimum governance-like tokens: {ADVISORY_MIN_TOKENS_GOV}")
    st.write(f"Advisory minimum strategy or partnerships-like tokens: {ADVISORY_MIN_TOKENS_STRAT_OR_PART}")

if uploaded is None:
    st.info("Please upload a PDF to begin.")
    st.stop()

pdf_bytes = io.BytesIO(uploaded.read())

st.info("Parsing PDF")
try:
    pages = extract_pages_from_pdf(pdf_bytes)
except Exception as e:
    st.error(f"Cannot parse PDF. {e}")
    st.stop()

# semantic sectioning
st.info("Finding governance, strategy, and partnerships content")
sectioner = SemanticSectioner(sim_threshold=0.34)
sections = sectioner.split(pages)

# advisory warnings only
total_tokens = sum(token_count(v) for v in sections.values())
gov_tokens = token_count(sections.get("governance", []))
strat_tokens = token_count(sections.get("strategy", []))
part_tokens = token_count(sections.get("partnerships", []))
if total_tokens < ADVISORY_MIN_WORDS_TOTAL:
    st.warning(f"Low total text volume detected, {total_tokens} tokens. Interpret results cautiously.")
if gov_tokens < ADVISORY_MIN_TOKENS_GOV:
    st.warning(f"Governance like content appears limited, {gov_tokens} tokens.")
if (strat_tokens + part_tokens) < ADVISORY_MIN_TOKENS_STRAT_OR_PART:
    st.warning(f"Strategy or partnerships like content appears limited, {strat_tokens + part_tokens} tokens.")

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

with st.expander("Semantic section coverage (tokens)"):
    cov = {k: token_count(v) for k, v in sections.items()}
    st.json(cov)

st.caption("Free MVP using only the uploaded PDF. No external enrichment. Extend lexicons and thresholds per sector for better performance.")
