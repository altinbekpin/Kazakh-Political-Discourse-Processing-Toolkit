# analyzer/discourse/onto_actor_topic.py
from typing import List, Dict
import re
from .onto_runtime import load_graph, build_label_index, _labels

def find_mentions(text: str, max_ngram: int = 5) -> List[Dict]:
    """Онтология label-деріне дәл келетін 1..5-сөздік терезелерді іздеу."""
    g = load_graph()
    idx = build_label_index()
    tokens = re.findall(r"[^\s]+(?:\s+[^\s]+){0,%d}" % (max_ngram-1), text)
    hits = []
    for chunk in tokens:
        key = " ".join(chunk.split()).lower()
        iris = idx.get(key)
        if not iris:
            continue
        start = text.find(chunk)
        if start < 0:
            continue
        for iri in iris:
            hits.append({
                "match": chunk,
                "span": [start, start+len(chunk)],
                "iri": str(iri),
                "labels": _labels(g, iri),
            })
    return hits

def tag_actor_topic(hits: List[Dict]) -> List[Dict]:
    """Эвристикамен түр беру: actor/org/topic (қаласаңыз OWL класстарын қосыңыз)."""
    typed = []
    for h in hits:
        labtxt = " ".join(h.get("labels", [])).lower()
        if any(k in labtxt for k in ["партия", "фракция", "ұйым", "министрлік", "министерство"]):
            h["kind"] = "org"
        elif any(k in labtxt for k in ["салық", "білім", "денсаулық", "инфляция", "экономика", "әлеумет"]):
            h["kind"] = "topic"
        else:
            h["kind"] = "actor"
        typed.append(h)
    return typed