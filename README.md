
# AugemenText - Text Augementation Service for NLP
All models are stored in /trained_models/
New models can be saved in that directory, and update translator_config.json accordingly.

## Features
- Utilizing NMT models from [huggingface.co](https://huggingface.co/models), trained by [Helsinky-NLP](https://github.com/Helsinki-NLP)
- Cosine Similarity to clean bad augmentations (can be overriden with custom metric)

## Installation
```bash
git clone repository
cd repository
python3 setup.py bdist_wheel
pip3 install /dist/text_augmentation-0.0.1-py3-none-any.whl
``` 
## Example

```python
import pandas as pd
from src.text_augmentation import AugmenText


if __name__ == '__main__':
    
    TRANSLATION_URL = "http://0.0.0.0:8000"  # address of 'Miriam' translation service
    
    query_list_heb = [
        "חנות המכולת של סלים",
        "אתמול בבוקר יוסי ומירב הלכו לבית הספר. שם הם פגשו את חבריהם לאחר שלא ראו אותם הרבה זמן עקב נגיף הקורונה"
    ]

    query_df_heb = pd.DataFrame({
        "text": query_list_heb,
        "label": -1,
        "dummy_1": 0,
        "dummy_2": 2
    })

    target_langs = [
        "fi",
        "de",
        "arb",
        "sv",
        
    ]

    ta = AugmenText(
        src_lang="heb",
        target_langs=target_langs,
        special_tokens=None,
        translation_url=TRANSLATION_URL
    )

    augmentations_heb = ta.augment_text(query_df_heb, similiarity_check=True, keep_score_threshold=0.1)

```
will yield the following data-augmented dataframe:

|    | text                                                   |   label |   dummy_1 |   dummy_2 | lang_aug   |   backtranslation_score |
|---:|:-------------------------------------------------------|--------:|----------:|----------:|:-----------|------------------------:|
|  0 | חנות המכולת של סלים                                    |      -1 |         0 |         2 | src        |                1        |
|  0 | "המכולת של סלים."                                      |      -1 |         0 |         2 | arb        |                0.814796 |
|  0 |  חנויות המזון של סלים                                 |      -1 |         0 |         2 | de         |                0.281021 |
|  0 | חנות כלי-הקיבול של סלים                              |      -1 |         0 |         2 | sv         |                0.611097 |
|  1 | אתמול הלכתי עם משפחתי לקניון לקנות נעליים חדשות        |      -1 |         0 |         2 | src        |                1        |
|  1 | ואז משפחתי ואני הלכנו לקניון לקנות נעליים חדשות.       |      -1 |         0 |         2 | arb        |                0.385984 |
|  1 | הלכתי לקניון עם משפחתי לקנות נעליים חדשות.             |      -1 |         0 |         2 | de         |                0.829918 |
|  1 |  המשפחה שלי ואני הלכנו לקניון כדי לקנות נעליים חדשות. |      -1 |         0 |         2 | fi         |                0.241974 |
|  1 |  הלכתי עם משפחתי לקנות נעליים חדשות                   |      -1 |         0 |         2 | sv         |                0.7681   |
