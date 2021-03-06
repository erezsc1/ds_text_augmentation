import pandas as pd
from src.text_augmentation import AugmenText


if __name__ == '__main__':

    TRANSLATION_URL = "http://0.0.0.0:8000"

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

    print(augmentations_heb.to_markdown())


    query_arb = [
        "محطة بنزين شعار هعيمق",
        "ذهب يوسي وميراف صباح أمس إلى المدرسة. هناك التقوا بأصدقائهم بعد أن لم يروهم لفترة طويلة بسبب كورونا"
    ]

    src_lang = "arb"

    target_langs = [
        "heb",
        "eng",
        "rus",
        "spa",
        "tur",
        "it",
        "pl",
        "el",
        "eo",
        "fr"
    ]

    query_arb_df = pd.DataFrame({
        "text": query_arb,
        "label": -1,
        "dummy_1": 0,
        "dummy_2": 2
    })


    ta = AugmenText("arb", target_langs, translation_url=TRANSLATION_URL)
    augmentations_arb = ta.augment_text(query_arb_df,similiarity_check=True,keep_score_threshold=0.2)

    print("Augmentations")
    print("# -------- HEB -------- #")
    pprint.pprint(augmentations_heb)

    print("# -------- ARB -------- #")
    pprint.pprint(augmentations_arb)

    print()
