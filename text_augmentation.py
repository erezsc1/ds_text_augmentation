import string
import pprint
import warnings
import pandas as pd
from translator import Translator
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

class AugmenText:
    def __init__(self, src_lang, target_langs, special_tokens=None):
        self.src_lang = src_lang
        self.target_langs = target_langs
        self.special_tokens = special_tokens

    def augment_text(
            self,
            query_df: pd.DataFrame,
            similiarity_check=False,
            keep_score_threshold: float=0.1
    ):
        '''
        :param query_list:
        :param similiarity_check:
        :param keep_score_threshold:
        :return:
        '''
        augmentations_df = self.augment(query_df)
        if similiarity_check:
            augmentations_df = augmentations_df[augmentations_df["backtranslation_score"] >= keep_score_threshold]
        return augmentations_df

    def augment(self, query_df : pd.DataFrame):
        '''
        :param query_df: pd.DataFrame - must include "text" column
        :return: pd.DataFrame - text augmented dataframe
        '''
        assert "text" in query_df.columns
        all_augmentations = query_df
        query_df["lang_aug"] = "src"
        for lang in self.target_langs:
            try:
                translator_front = Translator(source_lang=self.src_lang, target_lang=lang, special_tokens=self.special_tokens)
            except NotImplementedError:
                print(f"translation {self.src_lang} --> {lang} does not exist.")
                continue
            try:
                translator_back = Translator(source_lang=lang, target_lang=self.src_lang, special_tokens=self.special_tokens)
            except NotImplementedError:
                print(f"translation {lang} --> {self.src_lang} does not exist.")
                continue
            trans_text = translator_front.translate(query_df["text"])
            aug_text = translator_back.translate(trans_text["tgt_text"])
            cur_augmentations = pd.DataFrame(
                columns=[
                    "lang_aug",
                    "text",
                    "backtranslation_score"
                ]
            )
            cur_augmentations["text"] = aug_text["tgt_text"]
            cur_augmentations["lang_aug"] = lang

            del translator_front
            del translator_back

            all_augmentations = pd.concat([all_augmentations, cur_augmentations])

        all_augmentations.sort_index(inplace=True)

        unique_index_list = list(set(all_augmentations.index))
        for index in unique_index_list:
            # calc backtranslation scores
            temp_df = all_augmentations[all_augmentations.index == index]
            scores = self.calc_augmentation_score(
                temp_df["text"].tolist()
            )
            temp_df["backtranslation_score"] = scores
            #  fillna values of index
            cols_to_fill = set(temp_df.columns).difference(set(["lang_aug", "text", "backtranslation_score"]))
            for column in cols_to_fill:
                value = temp_df[temp_df["lang_aug"] == "src"][column].values[0]
                temp_df[column].fillna(value, inplace=True)

            all_augmentations[all_augmentations.index == index] = temp_df

        return all_augmentations

    def calc_augmentation_score(
            self,
            augmentations_list: list
        ):
        '''
        :param augmentations:  list[str] - assumes orignal text is first in the list
        :return: list
        '''
        trans_table = str.maketrans(" ", " ", string.punctuation)
        augmentation_query = [x.translate(trans_table) for x in augmentations_list]
        sim_vectorizer = TfidfVectorizer().fit_transform(augmentation_query)
        pairwise_similarity = sim_vectorizer * sim_vectorizer.T
        backtranslation_scores = list(pairwise_similarity.A[:, 0]) # discard the self similarity to query
        return backtranslation_scores

