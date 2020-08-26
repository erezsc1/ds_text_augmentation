
# AugemenText - Text Augementation Service for NLP




All models are stored in /trained_models/
New models can be saved in that directory, and update translator_config.json accordingly.
## Examples
### library calls
```python
from translator import Translator


if __name__ == '__main__':
    # heb -> arb
    translator = Translator("heb","arb")
    seq = ["שלום לכולם", "ארבעים ושתיים", "ארבעים וחמש"]

    result = translator.translate(seq)
```
### Client Class
```python
from translation_client import TranslatorClient


if __name__ == '__main__':
    service_url = "http://0.0.0.0:80"
    tc = TranslatorClient("heb", "arb", service_url)
    query1 = "שלום לכם, ילדים וילדות"
    query2 = [
        "שלום לכם, ילדים וילדות",
        "אני יובל המבולבל"
    ]
    tc.translate(query1)  # مرحباً أيها الأطفال والبنات
    tc.translate(query2)  # ['مرحباً أيها الأطفال والبنات', 'أنا يوبيل مشوّش']

```
