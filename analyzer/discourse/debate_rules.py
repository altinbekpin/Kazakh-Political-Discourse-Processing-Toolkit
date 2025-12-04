# analyzer/discourse/debate_rules.py
import re
from typing import List, Dict
from functools import lru_cache
from .onto_actor_topic import find_mentions, tag_actor_topic

SPEAKER_LINE = re.compile(r"^\s*([A-ZӘӨҰҮҚҒІА-Я][^\:]{0,60})\:\s*(.+)$", re.U)  # "Спикер: сөз..."
DASH_DIALOG  = re.compile(r"^\s*[-—]\s*(.+)$", re.U)

@lru_cache(maxsize=1)
def _compiled_lex():
    flags = re.I | re.U
    return {
        "attack": re.compile(r"\b(жоқ|дұрыс емес|өтірік|саңдырақ|қате)\b", flags),
        "defend": re.compile(r"\b(емес|жоқпыз|түсіндіріп өтейін|нақтылайын)\b", flags),
        "concede": re.compile(r"\b(иә,\s*бірақ|келісемін,\s*алайда|рас,\s*дегенмен)\b", flags),
        "question": re.compile(r"\?\s*$", flags),
    }

def segment_debate(text: str) -> List[Dict]:
    """
    Екі форматты қолдаймыз:
    1) "Аты-жөн: реплика"
    2) "— реплика" (спикер аты онтологиядан/контексттен анықталуы мүмкін; қазір unknown)
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    turns = []
    for ln in lines:
        m = SPEAKER_LINE.match(ln)
        if m:
            turns.append({"speaker": m.group(1).strip(), "text": m.group(2).strip()})
            continue
        m2 = DASH_DIALOG.match(ln)
        if m2:
            turns.append({"speaker": "unknown", "text": m2.group(1).strip()})
            continue
        # егер формат толық емес болса — бір блокқа жинай саламыз
        if turns and turns[-1]["speaker"] == "unknown":
            turns[-1]["text"] += " " + ln
        else:
            turns.append({"speaker": "unknown", "text": ln})
    return turns

def analyze_turn(turn: Dict) -> Dict:
    rx = _compiled_lex()
    txt = turn["text"]
    lab = "бейтарап"; score = 0.5
    # қарапайым эвристика:
    atk = bool(rx["attack"].search(txt))
    qst = bool(rx["question"].search(txt))
    dfd = bool(rx["defend"].search(txt))
    ccd = bool(rx["concede"].search(txt))
    if atk:    lab, score = "теріс", 0.72
    elif ccd:  lab, score = "оң",    0.66  # жұмсақ позитив
    elif dfd:  lab, score = "бейтарап", 0.55
    # адресат/нысана: онтологияға сүйеніп кім туралы айтылғанын табамыз
    onto = tag_actor_topic(find_mentions(txt))
    return {
        "speaker": turn["speaker"],
        "text": txt,
        "sentiment": lab, "score": score,
        "flags": {"attack": atk, "question": qst, "defend": dfd, "concede": ccd},
        "mentions": onto,   # actor/org/topic
    }

def analyze_debate(text: str) -> Dict:
    turns = segment_debate(text)
    results = [analyze_turn(t) for t in turns]
    # Спикерге жинақтау
    by_speaker: Dict[str, Dict[str,int]] = {}
    for r in results:
        sp = r["speaker"]
        dd = by_speaker.setdefault(sp, {"оң":0,"теріс":0,"бейтарап":0})
        dd[r["sentiment"]] += 1
    return {"turns": results, "by_speaker": by_speaker}