# analyzer/discourse/speech_rules.py
import re
from typing import List, Dict, Tuple
from functools import lru_cache
from  .onto_actor_topic import find_mentions, tag_actor_topic
from .campaign_onto_rules import detect_regex

# Бағалау лексикасы (қысқа тізім; негізгі тізім JSON-нан жүктеледі)
POS_VERBS = r"(қолдаймын|мақұлдаймын|құптаймын|мақтаймын|қажет деп санаймын|сенемін)"
NEG_VERBS = r"(сынға аламын|қарсымын|қабылдай алмаймын|жақтамаймын|жоққа шығарамын)"
POS_ADJ   = r"(дұрыс|орынды|маңызды|тиімді|әділетті)"
NEG_ADJ   = r"(дұрыс емес|ысырапшыл|ақымақ|қатерлі|әділетсіз)"

FIRST_PERSON = r"\b(мен|біз)\b"
SECOND_PERSON= r"\b(сіз|сіздер)\b"

@lru_cache(maxsize=1)
def _compiled():
    flags = re.I | re.U
    return {
        "pos_verb": re.compile(POS_VERBS, flags),
        "neg_verb": re.compile(NEG_VERBS, flags),
        "pos_adj":  re.compile(POS_ADJ, flags),
        "neg_adj":  re.compile(NEG_ADJ, flags),
        "i_pron":   re.compile(FIRST_PERSON, flags),
        "you_pron": re.compile(SECOND_PERSON, flags),
    }

def _sent_tokenize(text: str) -> List[Tuple[int,int,str]]:
    # Өте қарапайым бөлгіш: .!? бойынша, тырнақшаны елемей
    import re
    spans, start = [], 0
    for m in re.finditer(r"[.!?]+", text):
        end = m.end()
        chunk = text[start:end].strip()
        if chunk:
            spans.append((start, end, chunk))
        start = end
    tail = text[start:].strip()
    if tail:
        spans.append((start, start+len(tail), tail))
    return spans

def analyze_speech(text: str) -> Dict:
    rx = _compiled()
    # 1) Онтологиядан кандидат-нысана (actor/org/topic) іздейміз
    onto_hits = tag_actor_topic(find_mentions(text))
    # 2) Сөйлем бойынша жүріп, әр сөйлемге бағалау белгісін қоямыз
    sent_spans = _sent_tokenize(text)
    items = []
    for (s,e,chunk) in sent_spans:
        
        lab = "бейтарап"; score = 0.5
        rx_hits_sent = [h for h in detect_regex(chunk)]
        rx_devices = {h["device"] for h in rx_hits_sent}

        # Егер жарнамалық әдістер табылса, спикер сөзі үшін де оны оң белгі ретінде ескереміз
        if {"intent_or_plan", "promise", "call_to_action"} & rx_devices:
            lab, score = "оң", max(0.70, 0.70 if 'call_to_action' in rx_devices else 0.66)

        # Егер шабуыл болса — теріс
        if "attack_ad" in rx_devices:
            lab, score = "теріс", max(score, 0.72)

        # Рәсімдік болса — бейтарапқа жұмсартады
        if "procedural" in rx_devices and lab == "бейтарап":
            lab, score = "бейтарап", min(score, 0.60)
        pos = int(bool(rx["pos_verb"].search(chunk) or rx["pos_adj"].search(chunk)))
        neg = int(bool(rx["neg_verb"].search(chunk) or rx["neg_adj"].search(chunk)))
        if pos and not neg:
            lab, score = "оң", 0.70
        elif neg and not pos:
            lab, score = "теріс", 0.72
        # адресат: егер сөйлемде 2-жақ болса – қарсы жаққа бағытталған деп шамалаймыз
        target = "unspecified"
        if rx["you_pron"].search(chunk):
            target = "opponent"
        # сөйлемнің ішіндегі онтология сәйкестіктерін байлап жібереміз
        local_refs = [h for h in onto_hits if h["span"][0] >= s and h["span"][1] <= e]
        items.append({
            "span": [s,e], "text": chunk,
            "sentiment": lab, "score": score,
            "targets": local_refs,    # actor/org/topic қатар
            "address": target,        # opponent/self/unspecified
        })
    # 3) Нысанаға жинақтау: actor/org/topic бойынша басым баға
    stance = {}
    for it in items:
        tgs = it["targets"] or []
        if not tgs:
            continue
        for t in tgs:
            key = (t["kind"], t["iri"])
            arr = stance.setdefault(key, [])
            arr.append(it["sentiment"])
    stance_out = []
    for (kind, iri), arr in stance.items():
        pos = arr.count("оң"); neg = arr.count("теріс")
        lab = "бейтарап"
        if pos>neg: lab="оң"
        elif neg>pos: lab="теріс"
        stance_out.append({"kind":kind, "iri":iri, "label":lab, "votes":{"pos":pos,"neg":neg}})
    return {
        "items": items,          # сөйлемдік баға және нысана
        "stance": stance_out,    # нысанаға қатысты жинақталған баға
        "onto_hits": onto_hits,  # жалпы сәйкестіктер
    }