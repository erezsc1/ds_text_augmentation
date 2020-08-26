import pandas as pd
from src.text_augmentation import AugmenText

if __name__ == '__main__':
    TRANSLATION_URL = "http://0.0.0.0:8000"  # address of 'Miriam' translation service

    query_list_heb = [
        "חנות המכולת של סלים",
        "אתמול הלכתי עם משפחתי לקניון לקנות נעליים חדשות"
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

    print(augmentations_heb.to_markdown())