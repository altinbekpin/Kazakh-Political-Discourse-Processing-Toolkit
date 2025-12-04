# Қазақ/орыс/ағылшынша синонимдер – predicates & classes атауларын табуға көмектеседі
PROPERTY_LABEL_CANDS = {
    "hasDevice":      {"hasDevice","device","hasRhetoricType","hasRhetoric","риторикалық әдіс","риторический прием"},
    "hasPolarity":    {"hasPolarity","polarity","sentiment","полярность","баға","бағалау"},
    "hasDomain":      {"hasDomain","domain","домен","сала","пәндік сала"},
    "hasPattern":     {"hasPattern","pattern","regex","үлгі","өрнек"},
    "expands":        {"expands","mapsTo","expansion","жазылуы","толық атауы"},
    "hasLabelProp":   {"label","name","title"},  # егер rdfs:label/skos:* жоқ болса
}

CLASS_LABEL_CANDS = {
    "RhetoricDevice":     {"RhetoricDevice","риторикалық әдіс","риторический прием"},
    "MWE":                {"MWE","multiword","көпсөзді тіркес"},
    "Abbreviation":       {"Abbreviation","аббревиатура","қысқартуы"},
    "ElectionTerm":       {"ElectionTerm","сайлау термины","сайлауалды термин"},
    "SentimentCategory":  {"Sentiment","SentimentCategory","сентимент"},
    "EmotionCategory":    {"Emotion","EmotionCategory","эмоция"},
    "Domain":             {"Domain","Домен","Пәндік сала"},
}
# Онтология тіл тэгтері
LANGS = {"kk","kaz","ru","en", None}