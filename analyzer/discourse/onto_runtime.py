# analyzer/discourse/onto_runtime.py
import os, re
from pathlib import Path
from functools import lru_cache
from typing import List, Dict
from rdflib import Graph, URIRef, Literal, RDFS
from rdflib.namespace import SKOS

ONTO_PATH = Path(os.getenv("POLISENT_ONTO_PATH", "data/political_discourse_ontology_final.owl"))
LANGS = {"kk", "kaz", "ru", "en", None}

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()

@lru_cache(maxsize=1)
def load_graph() -> Graph:
    if not ONTO_PATH.exists():
        raise FileNotFoundError(f"Ontology not found: {ONTO_PATH}")
    g = Graph()
    g.parse(str(ONTO_PATH))
    return g

def _labels(g: Graph, node: URIRef) -> List[str]:
    out = []
    for p in (SKOS.prefLabel, RDFS.label, SKOS.altLabel):
        for o in g.objects(node, p):
            if isinstance(o, Literal) and (o.language in LANGS):
                out.append(str(o))
    if not out:
        try:
            out.append(node.split("#")[-1].split("/")[-1])
        except Exception:
            out.append(str(node))
    # unique
    seen, res = set(), []
    for v in out:
        if v not in seen:
            seen.add(v); res.append(v)
    return res

@lru_cache(maxsize=1)
def build_label_index() -> Dict[str, List[URIRef]]:
    g = load_graph()
    idx: Dict[str, List[URIRef]] = {}
    for s, p, o in g:
        if p in (RDFS.label, SKOS.prefLabel, SKOS.altLabel) and isinstance(o, Literal) and (o.language in LANGS):
            idx.setdefault(_norm(str(o)), []).append(s)
    return idx

def match_text_with_ontology(text: str) -> List[dict]:
    """Label-ге дәл келетін 1–5 сөздік терезелерді іздеу (қысқа нұсқа)."""
    g = load_graph()
    hits = []
    tokens = re.findall(r"[^\s]+(?:\s+[^\s]+){0,4}", text)  # 1..5 сөз
    for chunk in tokens:
        key = _norm(chunk)
        iris = build_label_index().get(key, [])
        if not iris:
            continue
        start = text.find(chunk)
        if start < 0:
            continue
        for iri in iris:
            hits.append({
                "match": chunk,
                "span": [start, start+len(chunk)],
                "inst": str(iri),
                "labels": _labels(g, iri),
                "onto_devices": [],
                "onto_polarity": None,
            })
    return hits