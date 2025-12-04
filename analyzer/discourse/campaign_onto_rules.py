import re
from functools import lru_cache
from typing import List, Dict
from .onto_runtime import match_text_with_ontology

# Қысқа домендік regex триггерлер (ереже)
REGEX_RULES = {
    "қолдауға_шақыру": [
        r"\bдауыс\s+бер(іңіз|іңіздер|ейік)\b",
        r"\bқолда(ңыз|ңыздар|уға)\b",
        r"\bқосыл(ыңыз|ыңыздар|уға)\b",
    ],
    "жоспар": [
        r"\b(мен|біз)\s+депутат( болсам| болсақ)\b",
        r"\b(болсам|болсақ)\b.*\b(жасаймын|етемін|өзгертемін|қайта қараймын|алып тастаймын)\b",
        r"\b(жоспарлаймын|көздеймін|ниеттенемін)\b",
    ],
    "уәде": [
        r"\bуәде\s+бер(еміз|емін)\b",
        r"\bжүзеге\s+асырамыз\b|\bорындаймыз\b",
        r"\bжалақы(ны)?\s+көтереміз\b",
    # 1-жақ келер шақ маркерлері
        r"\b(жасаймын|етемін|орындаймын|жүзеге асырамын|әзірлеймін)\b",
        r"\b(азайтамын|жоғартамын|тоқтатамын|қысқартамын|қараймын|қайта қараймын|алып тастаймын)\b",
    # көпше 1-жақ та пайдалы
    r"\b(жасаймыз|етеміз|орындаймыз|жүзеге асырамыз|көтереміз|қысқартамыз|алып тастаймыз)\b",
    ],
    "шабуылдау": [
        r"\bөтірік\b|\bжалған\b|\bмасқара\b|\bжемқор(лық)?\b",
        r"\bуәдесін\s+орындама(ды|ған)\b",
    ],
    "процедура": [
        r"\bүгіт[-\s]?насихат\s+кезең(і|інің)\b",
        r"\bорталық\s+сайлау\s+комиссиясы\b|\bОСК\b",
        r"\bдауыс\s+беру\s+нәтижесі\b",
    ],
}

@lru_cache(maxsize=1)
def _compiled():
    import re
    flags = re.I | re.U
    return {k:[re.compile(p, flags) for p in ps] for k,ps in REGEX_RULES.items()}

def detect_regex(text: str) -> List[dict]:
    hits = []
    for device, regs in _compiled().items():
        for rx in regs:
            for m in rx.finditer(text):
                hits.append({
                    "device": device,
                    "pattern": rx.pattern,
                    "match": m.group(0),
                    "span": [m.start(), m.end()],
                })
    return hits

def summarize_devices(hits: List[dict], onto_hits: List[dict]) -> Dict[str,int]:
    cnt: Dict[str,int] = {}
    for h in hits:
        cnt[h["device"]] = cnt.get(h["device"], 0) + 1
    # Онтологиядағы device label-дарын да санаймыз
    for oh in onto_hits:
        for d in oh.get("onto_devices", []):
            cnt[d] = cnt.get(d, 0) + 1
    return cnt

def rule_sentiment_emotion(dev_count: Dict[str,int]) -> Dict[str,str]:
    if dev_count.get("шабуыл",0) > 0:
        return {"sentiment":"теріс","emotion":"ашу"}
    if (dev_count.get("қолдауға_шақыру",0) > 0 or
        dev_count.get("уәде",0) > 0 or
        dev_count.get("жоспар",0) > 0):
        return {"sentiment":"оң","emotion":"сенім"}
    if dev_count.get("процедура",0) > 0:
        return {"sentiment":"бейтарап","emotion":"бейтарап"}
    return {"sentiment":"бейтарап","emotion":"бейтарап"}

def polarity_prior_from_onto(onto_hits: List[dict]) -> str|None:
    pols = [h.get("onto_polarity") for h in onto_hits if h.get("onto_polarity")]
    if not pols: return None
    if "теріс" in pols: return "теріс"
    if "оң" in pols:    return "оң"
    if "бейтарап" in pols: return "бейтарап"
    return None

def detect_campaign_with_ontology(text: str) -> dict:
    """EREJE + ONTO: матчтар, device санағы, полярлық приор."""
    rx_hits   = detect_regex(text)
    onto_hits = match_text_with_ontology(text)
    dev_count = summarize_devices(rx_hits, onto_hits)
    rule_pred = rule_sentiment_emotion(dev_count)
    onto_prior = polarity_prior_from_onto(onto_hits)
    return {
        "rx_hits": rx_hits,
        "onto_hits": onto_hits,
        "devices": dev_count,
        "rule_pred": rule_pred,
        "onto_prior": onto_prior,
    }

def post_fusion(base_label: str, base_score: float, pack: dict) -> tuple[str,float]:
    """Нейро/фолбэк нәтижесін домен ережелері мен онтология приорымен түзету."""
    devs = pack["devices"]
    # Процедура басым – бейтарапқа сырғыту
    if devs.get("процедура",0) >= max(1, devs.get("шабуыл",0)):
        base_label, base_score = "бейтарап", min(base_score, 0.60)
    # Шабуыл – теріс
    if devs.get("шабуыл",0) > 0 and base_label != "теріс":
        base_label, base_score = "теріс", max(base_score, 0.72)
    # Уәде/шақыру – оң
    if (devs.get("қолдауға_шақыру",0) + devs.get("уәде",0) + devs.get("жоспар",0)) > 0 \
    and base_label != "оң":
        base_label, base_score = "оң", max(base_score, 0.70)

    # Онтология приоры (күшті сигнал)
    op = pack.get("onto_prior")
    if op == "теріс" and base_label != "теріс":
        base_label, base_score = "теріс", max(base_score, 0.74)
    elif op == "оң" and base_label != "оң":
        base_label, base_score = "оң", max(base_score, 0.72)
    elif op == "бейтарап":
        base_label, base_score = "бейтарап", min(base_score, 0.60)

    return base_label, base_score