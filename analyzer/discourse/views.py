from django.shortcuts import render
from .forms import AnalyzerForm
from functools import lru_cache
import re

from .speech_rules import analyze_speech
from .debate_rules import analyze_debate

from transformers import pipeline, AutoTokenizer
import os
# файлдың басында:
import logging, traceback
from django.conf import settings

logger = logging.getLogger(__name__)
# --- ADDED: саяси дискурс модулі және жолдар
from django.conf import settings  # ADDED
from pathlib import Path          # ADDED
from .pd_analyzer import (        # ADDED (салыстырмалы импорт!)
    build_index_from_json,
    score_text as score_pd_text,
    is_political as is_pd
)

PD_JSON_PATH = os.getenv(         # ADDED: JSON жолы (қалауыңызша өзгертіңіз)
    "PD_JSON_PATH",
    str(Path(getattr(settings, "BASE_DIR", ".")) / "data" / "political_discourse_terms.json")
)
@lru_cache(maxsize=1)             # ADDED: индекс бір рет жүктелсін
def get_pd_index():
    return build_index_from_json(PD_JSON_PATH)

EMOTIONS = ["қуаныш","ашу","қорқыныш","жеккөрушілік","таңдану","қайғы","сенім","бейтарап"]
SENTIMENT = ["оң","бейтарап","теріс"]
MODEL = "joeddav/xlm-roberta-large-xnli"

# --- өте жеңіл fallback (ML жоқта жұмыс істейді)
POS = {"жақсы","тамаша","ұнамды","сенемін","қолдаймын","артықшылық"}
NEG = {"жаман","нашар","қате","өтірік","ұнамайды","қарсымын","жемқор","қорқынышты"}
EMO_MAP = {
    "ашу": {"ыза","ашу","ашулы","ренж","ұрыс"},
    "қорқыныш":{"қорқыныш","үрей","қауіп"},
    "қуаныш":{"қуаныш","қуанды","қуан","мақтаныш"},
    "жеккөрушілік":{"жек көр","жиіркен","масқара"},
    "қайғы":{"қайғы","мұң","реніш"},
    "сенім":{"сенім","үміт","үміттен"},
    "таңдану":{"таң","таңғал","таңғалды"},
}
# --- ADD: imports (файлдың бас жағында)
import json
from django.conf import settings
from pathlib import Path

# JSON: D==1 сүзгісімен жасалған сөздік тұрған жол (қажет болса ауыстырыңыз)
PD_JSON_PATH = os.getenv(
    "PD_JSON_PATH",
    str(Path(getattr(settings, "BASE_DIR", ".")) / "data" / "political_discourse_terms.json")
)

from functools import lru_cache

@lru_cache(maxsize=1)
def get_tactic_labels():
    """JSON кілттері – тактика атаулары"""
    with open(PD_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Тактика атаулары (sheet аттары)
    return [str(k) for k in data.keys() if k and str(k).strip()]

def classify_tactic_zero_shot(text: str):
    """
    Zero-shot арқылы тактиканы анықтау.
    MODEL/PIPELINE сіздегі get_clf() арқылы жүктеледі (негізгі кодқа тимейміз).
    """
    labels = get_tactic_labels()
    if not labels:
        return {"label": "", "score": 0.0, "alternatives": [], "warning": "Тактика тізімі бос"}

    # НАЗАР: zero-shot үшін гипотезада {label} плейсхолдері болуы керек
    hypothesis = "Бұл мәтінде {} тактикасы қолданылған."

    try:
        clf = get_clf()  # сіздің бар pipeline
        out = clf(text, labels, hypothesis_template=hypothesis, multi_label=False)
        return {
            "label": out["labels"][0],
            "score": float(out["scores"][0]),
            "alternatives": [
                {"label": l, "score": float(s)} for l, s in zip(out["labels"][1:5], out["scores"][1:5])
            ],
            "warning": None,
        }
    except Exception as e:
        return {"label": "", "score": 0.0, "alternatives": [], "warning": f"ZS қате: {e}"}

def _fallback(text, task):
    t = text.lower()
    words = set(re.findall(r"\w+", t))
    p, n = len(POS & words), len(NEG & words)
    sent = "бейтарап"
    if n > p: sent = "теріс"
    elif p > n: sent = "оң"

    best_emo, best_cnt = "бейтарап", 0
    for emo, lex in EMO_MAP.items():
        cnt = sum(1 for w in lex if w in t)
        if cnt > best_cnt:
            best_cnt, best_emo = cnt, emo

    label = best_emo if task == "emotion" else sent
    score = float(min(0.95, 0.55 + 0.1*abs(p-n) + 0.1*best_cnt))
    alternatives = []
    return {
        "label": label,
        "score": score,
        "alternatives": alternatives,
        "warning": "Ескерту: ML модель қолжетімсіз (transformers/torch/protobuf/sentencepiece жоқ). Fallback қолданылды."
    }


MODEL_ID = "joeddav/xlm-roberta-large-xnli"   # модель басы
TOKENIZER_ID = os.getenv("POLISENT_TOKENIZER", "xlm-roberta-large")  # токенайзер базадан

@lru_cache(maxsize=1)
def _get_pipeline():
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)  # Fast tokenizer, sentencepiece керек емес
    try:
        import torch
        device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else -1
    except Exception:
        device = -1
    return pipeline("zero-shot-classification", model=MODEL_ID, tokenizer=tok, device=device)

from functools import lru_cache
from transformers import pipeline, AutoTokenizer
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "hf" / "xnli_mdl"
TOKENIZER_DIR = Path(__file__).resolve().parent.parent / "hf" / "xnli_tok"

@lru_cache(maxsize=1)
def get_clf():
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))  # use_fast=True by default; SP керек емес
    # CPU-ға мәжбүрлейміз — Mac MPS кейде оп-ларды қолдамай, құлатады
    device = -1   # CPU
    return pipeline("zero-shot-classification", model=str(MODEL_DIR), tokenizer=tok, device=device)
def _is_candidate_speech(d: str) -> bool:
    d = (d or "").lower()
    return d in {"candidate_speech", "speech", "саяси қайраткер сөзі", "үміткер сөзі"} or "speech" in d

def _is_debate(d: str) -> bool:
    d = (d or "").lower()
    return d in {"debate", "сайлауалды пікірсайыс"} or "debate" in d
def analyze_text(text, domain, source, task):
    """
    Негізгі анализ:
    - Zero-shot (немесе фолбэк) => label/score
    - Егер domain == campaign_ad болса:
        * regex + OWL (classes & facts) арқылы матчтар
        * онтология приоры (hasPolarity) және device санағы
        * fusion -> финалдық label/score (sentiment task-та)
    Қайтарым құрылымы алдыңғыдай, тек result["campaign"] қосылады.
    """
    # 1) Базалық жіктеу (zero-shot немесе фолбэк)
    labels = EMOTIONS if task == "emotion" else SENTIMENT
    hyp = f"Бұл {{}} екенін көрсетеді. Мәтін {source} дерек көзінен және '{domain}' доменінен."
    print(domain)
    try:
        clf = get_clf()  # кэштелген pipeline
        out = clf(text, labels, hypothesis_template=hyp, multi_label=False)
        base = {
            "label": out["labels"][0],
            "score": float(out["scores"][0]),
            "alternatives": [
                {"label": l, "score": float(s)}
                for l, s in zip(out["labels"][1:4], out["scores"][1:4])
            ],
            "warning": None,
        }
    except Exception as e:
        print("ML load/infer error:", repr(e))
        base = _fallback(text, task)  # {"label","score","alternatives","warning"}

    if _is_candidate_speech(domain):
        sp = analyze_speech(text)
        base["speech"] = sp
        # Fusion/қорытынды: егер нысана бойынша жиынтық теріс/оң басым болса, жалпы реңкке әсер етеміз
        pos_votes = sum(1 for s in sp["stance"] if s["label"]=="оң")
        neg_votes = sum(1 for s in sp["stance"] if s["label"]=="теріс")
        if task != "emotion":
            if pos_votes>neg_votes and base["label"]!="оң":
                base["label"], base["score"] = "оң", max(base["score"], 0.68)
            elif neg_votes>pos_votes and base["label"]!="теріс":
                base["label"], base["score"] = "теріс", max(base["score"], 0.70)

    if _is_debate(domain):
        print(domain)
        db = analyze_debate(text)
        base["debate"] = db
        # Fusion/қорытынды: спикерлер бойынша теріс реплика саны көп болса — теріске тартады
        neg = sum(t["sentiment"]=="теріс" for t in db["turns"])
        pos = sum(t["sentiment"]=="оң" for t in db["turns"])
        if task != "emotion":
            if neg>pos and base["label"]!="теріс":
                base["label"], base["score"] = "теріс", max(base["score"], 0.70)
            elif pos>neg and base["label"]!="оң":
                base["label"], base["score"] = "оң", max(base["score"], 0.66)
    # 2) Сайлауалды жарнама домені үшін: онтология + ережелер + fusion
    def _is_campaign(d: str) -> bool:
        d = (d or "").lower().strip()
        return d in {"campaign_ad", "campaign", "сайлауалды жарнама"} or "campaign" in d
    if _is_campaign(domain):
        try:
            # Импортты осында ұстаймыз: модуль жоқ болса, graceful degrade
            from .campaign_onto_rules import (
                detect_campaign_with_ontology,  # regex + OWL hits (classes & facts)
                post_fusion,                    # fusion: rules + ontology prior
            )

            pack = detect_campaign_with_ontology(text)  # {'rx_hits','onto_hits','devices','rule_pred','onto_prior'}
            fused_label, fused_score = post_fusion(base["label"], base["score"], pack)
            # UI үшін дәлелдер мен есеп
            base["campaign"] = {
                "devices": pack.get("devices", {}),
                "rule_pred": pack.get("rule_pred", {}),
                "onto_prior": pack.get("onto_prior"),
                "hits": {
                    "regex": pack.get("rx_hits", []),
                    "ontology": pack.get("onto_hits", []),  # әр hit: inst/class/labels/onto_devices/onto_polarity/...
                },
                "fused": {"label": fused_label, "score": fused_score},
            }

            # Тек sentiment тапсырмасында финал ретінде fusion-ды қолданамыз (emotion – өзгеріссіз)
            if task != "emotion":
                base["label"], base["score"] = fused_label, fused_score

        except ImportError as ie:
            print(ie)
            # rdflib немесе модульдер орнатылмаған
            w = "Онтология интеграциясы сөнді: тәуелділік жоқ (rdflib/модуль)."
            base["warning"] = f"{base.get('warning') + '; ' if base.get('warning') else ''}{w}"
        except Exception as e:
            # Онтологияны оқу/парс қателері
            w = f"Онтология интеграция қателігі: {e}"
            base["warning"] = f"{base.get('warning') + '; ' if base.get('warning') else ''}{w}"

    return base

def analyzer_view(request):
    ctx = {"result": None, "error": None, "form": AnalyzerForm()}
    if request.method == "POST":
        form = AnalyzerForm(request.POST)
        ctx["form"] = form
        if form.is_valid():
            cd = form.cleaned_data
            base = analyze_text(cd["text"], cd["domain"], cd["source"], cd["task"])

            # --- ADD: Zero-shot арқылы тактика классификациясы
            base["tactic_zero_shot"] = classify_tactic_zero_shot(cd["text"])

            ctx["result"] = base
    return render(request, "form.html", ctx)

