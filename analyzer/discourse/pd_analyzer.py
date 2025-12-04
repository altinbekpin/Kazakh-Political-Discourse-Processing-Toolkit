import json, re

KAZ_LETTERS = "А-Яа-яЁёІіҢңӘәҒғҚқӨөҰұҮүҺһ"
TOKEN_RE = re.compile(f"[{KAZ_LETTERS}]+", re.UNICODE)

def norm(s: str) -> str:
    import re
    return re.sub(r"\s+", " ", (s or "").lower()).strip()

def is_phrase(term: str) -> bool:
    return " " in term.strip()

def build_index_from_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    idx = {}
    for tactic, items in data.items():
        phrases, tokens = set(), set()
        for it in items:
            name = norm(it.get("termin", {}).get("name", ""))
            if not name:
                continue
            if is_phrase(name):
                phrases.add(name)
            else:
                tokens.add(name)
        idx[tactic] = {"phrases": phrases, "tokens": tokens}
    return idx

def tokenize(text: str):
    return TOKEN_RE.findall(norm(text))

def _token_roots(token: str):
    """Very-light stemming: '...у' -> түбір ('қанау' -> 'қана')"""
    roots = {token}
    if token.endswith("у") and len(token) > 1:
        roots.add(token[:-1])
    return roots

# --- РЕЗЕРВТІК саяси cue түбірлер (JSON табылмаса іске қосамыз)
CUE_POL = {
    "саясат","партия","үкімет","мәжіліс","сенат","парламент","билік",
    "сайлау","үгіт","коалиц","жемқор","қана","оппозиц","халық","ел"
}

def _cue_fallback(toks):
    """JSON-нан ештеңе шықпаса, cue түбірлері бойынша ұпай есептейміз."""
    hits = []
    total = 0
    for root in CUE_POL:
        cnt = sum(1 for t in toks if t.startswith(root))
        if cnt > 0:
            hits.append((root, cnt, "cue"))
            total += cnt
    hits.sort(key=lambda x: x[1], reverse=True)
    # тек алғашқы 10 дәлелді қалдырамыз
    return total, hits[:10]

def score_text(text: str, index: dict, weight_phrase: float = 2.0, weight_token: float = 1.0):
    toks = tokenize(text)
    joined = " " + " ".join(toks) + " "
    results = []

    # 1) JSON-дағы тактикалар бойынша бағалау
    for tactic, bags in index.items():
        score = 0.0
        hits = []

        # Фразалар
        for ph in bags["phrases"]:
            if ph in joined:
                c = joined.count(ph)
                if c > 0:
                    score += weight_phrase * c
                    hits.append((ph, c, "phrase"))

        # Жалқы сөздер (+ түбір)
        for tk in bags["tokens"]:
            cnt = 0
            for r in _token_roots(tk):
                cnt += sum(1 for t in toks if t.startswith(r))
            if cnt > 0:
                score += weight_token * cnt
                hits.append((tk, cnt, "prefix"))

        if score > 0:
            hits_sorted = sorted(hits, key=lambda x: x[1], reverse=True)[:10]
            results.append((tactic, score, hits_sorted))

    # 2) Егер ештеңе табылмаса — cue fallback
    if not results:
        cscore, chits = _cue_fallback(toks)
        if cscore > 0:
            results.append(("CueWords", float(cscore), chits))

    results.sort(key=lambda x: x[1], reverse=True)
    return results

def is_political(results, min_top_score: float = 3.0, min_unique_terms: int = 2):
    if not results:
        return False
    top_score = results[0][1]
    if top_score >= min_top_score:
        return True
    uniq = set()
    for _, _, hits in results:
        for h in hits:
            uniq.add(h[0])
    return len(uniq) >= min_unique_terms