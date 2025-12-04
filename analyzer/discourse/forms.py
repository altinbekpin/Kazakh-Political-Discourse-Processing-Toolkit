from django import forms

DOMAIN_CHOICES = [
    ("саяси дискурс", "Саяси дискурс"),
    ("сайлауалды жарнама", "Сайлауалды жарнама"),
    ("үміткер сөзі", "Үміткер сөзі"),
    ("сайлауалды пікірсайыс", "Сайлауалды пікірсайыс"),
]
SOURCE_CHOICES = [("ресми", "Ресми"), ("бейресми", "Бейресми")]
TASK_CHOICES = [("emotion", "Эмоция"), ("sentiment", "Сентимент")]

class AnalyzerForm(forms.Form):
    text = forms.CharField(
        label="Мәтін",
        widget=forms.Textarea(attrs={"rows": 6, "placeholder": "Мәтінді осында жабыстырыңыз..."})
    )
    domain = forms.ChoiceField(label="Домен", choices=DOMAIN_CHOICES, initial="саяси дискурс")
    source = forms.ChoiceField(label="Дерек көзі", choices=SOURCE_CHOICES, initial="ресми")
    task = forms.ChoiceField(label="Тапсырма", choices=TASK_CHOICES, initial="emotion")