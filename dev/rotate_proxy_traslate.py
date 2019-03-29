import os
import time
import pandas as pd
import numpy as np
from googletrans import Translator
import textblob
import nltk
from textblob import TextBlob
import requests
from lxml.html import fromstring
from itertools import cycle

class GAPdf:

    def __init__(self):
        self.proxies = self.get_proxies()
        self.proxies = self.proxies + [
                # '122.50.6.186:80',
                # '181.129.41.163:8080',
                # '188.165.178.241:1080',
                # '151.106.10.62:8080',
                # '177.137.193.67:8080',
                # '95.79.57.206:53281',
                # '203.210.84.122:46964',
                # '177.91.127.32:54683',
                # '119.82.253.66:58918',
                # '103.43.40.122:30944',
                ]
        self.proxy_pool = cycle(self.proxies)
        # self.middle_lang = ['nl','fi','el','hi','it','la','pl','ru','es','sv','tr','ja','zn-CN']
        self.middle_lang = ['nl','fr','de','es']

        self.df_train = pd.read_csv("~/gender-pronoun/input/gap-test.tsv", delimiter="\t")
        self.df_val = pd.read_csv("~/gender-pronoun/input/gap-validation.tsv", delimiter="\t")
        self.df_test = pd.read_csv("~/gender-pronoun/input/gap-development.tsv", delimiter="\t")
        self.sample_sub = pd.read_csv("~/gender-pronoun/input/sample_submission_stage_1.csv")
        assert self.sample_sub.shape[0] == self.df_test.shape[0]

        self.df_test_trans = pd.read_csv("~/gender-pronoun/input/gap-test-trans.csv")
        self.df_val_trans = pd.read_csv("~/gender-pronoun/input/gap-validation-trans.csv")
        self.df_train_trans = pd.read_csv("~/gender-pronoun/input/gap-development-trans.csv")

        self.process_df()

    def process_df(self):
        self.df_train = self.extract_target(self.df_train).sample(n=8)
        self.df_val = self.extract_target(self.df_val).sample(n=2)
        self.df_test = self.extract_target(self.df_test).sample(n=8)
        self.df_train['Text_tag'] = self.df_train.apply(self.replace_tag,axis=1)
        self.df_val['Text_tag'] = self.df_val.apply(self.replace_tag,axis=1)
        self.df_test['Text_tag'] = self.df_test.apply(self.replace_tag,axis=1)

        self.df_train['Text_bt'] = self.df_train['Text_tag'].map(self.translate)
        self.df_val['Text_bt'] = self.df_val['Text_tag'].map(self.translate)
        self.df_test['Text_bt'] = self.df_test['Text_tag'].map(self.translate)

        self.df_train_trans = pd.concat([self.df_train_trans, self.df_train])
        self.df_val_trans = pd.concat([self.df_val_trans, self.df_val])
        self.df_test_trans = pd.concat([self.df_test_trans, self.df_test])

        self.df_train_trans.to_csv("~/gender-pronoun/input/gap-development-trans.csv")
        self.df_val_trans.to_csv("~/gender-pronoun/input/gap-validation-trans.csv")
        self.df_test_trans.to_csv("~/gender-pronoun/input/gap-test-trans.csv")

    def get_proxies(self):
        url = 'https://free-proxy-list.net/'
        response = requests.get(url)
        parser = fromstring(response.text)
        proxies = []
        for i in parser.xpath('//tbody/tr')[:20]:
            if i.xpath('.//td[7][contains(text(),"yes")]'):
                #Grabbing IP and corresponding PORT
                proxy = ":".join([i.xpath('.//td[1]/text()')[0], i.xpath('.//td[2]/text()')[0]])
                proxies.append(proxy)
        return proxies

    def translate(self, text):
        translator = Translator(timeout=10)
        target_text = text
        src_lang = 'en'
        dest_lang = 'en'
        trans_time=2
        count=0
        print(target_text)

        while True:
            # translator.session.proxies.update({'https':next(self.proxy_pool)})
            nltk.set_proxy('http://'+next(self.proxy_pool))

            while(src_lang==dest_lang):
                dest_lang = np.random.choice(self.middle_lang)
            if count==trans_time:
                dest_lang='en'

            try:
                blob=TextBlob(self.insert_protect_tag(target_text))
                # target_text = translator.translate(target_text,
                        # src=src_lang, dest=dest_lang).text
                target_text = self.remove_protect_tag(blob.translate(from_lang=src_lang,to=dest_lang).raw)
                print('src lang: '+src_lang)
                print('dest lang: '+dest_lang)
                print('trans text: '+target_text)
                src_lang = dest_lang
                if count == trans_time:
                    break
                count+=1
            except Exception as e:
                print("skip, connection error: "+str(e))

            time.sleep(1)
        return target_text

    def insert_protect_tag(self, text):
        return text

    def remove_protect_tag(self, text):
        return text

    def insert_protect_tag_span(self, text):
        start_tags = ['[[A-S]]','[[B-S]]','[[P-S]]']
        for t in start_tags:
            text = text.replace(t,"<span class='notranslate'>"+t)
        text = text.replace("-E]]","-E]]</span>")
        print(text)
        return text

    def remove_protect_tag_span(self, text):
        text = text.replace("<span class='notranslate'>","")
        text = text.replace("</span>","")
        print(text)
        return text

    def insert_protect_tag_arrow(self, text):
        start_tags = ['[[A]]','[[B]]','[[P]]']
        for t in start_tags:
            text = text.replace(t,"<< "+t)
        text = text.replace("-E]]","-E]] >>")
        print(text)
        return text

    def remove_protect_tag_arrow(self, text):
        text = text.replace("<< ","")
        text = text.replace(" >>","")
        print(text)
        return text

    def replace_tag(self, row):
        """
        replace to single token tag with extra chars, to avoid translate by translator
        """

        to_be_inserted = sorted([
            (row["A-offset"], "<< [[A", len(row["A"]), "]] >>", row["A"]),
            (row["B-offset"], "<< [[B", len(row["B"]), "]] >>", row["B"]),
            (row["Pronoun-offset"], "<< [[P", len(row["Pronoun"]), "]] >>", row["Pronoun"])
        ], key=lambda x: x[0], reverse=True)
        text = row["Text"]
        for offset, tag, l, etag, w in to_be_inserted:
            if text[offset+l:offset+l+2] == "'s":
                l += 2
                w = w+"'s"
            text = text[:offset] + tag + \
                    etag + \
                    text[offset+l:]
        return text

    def insert_tag(self, row):
        """Insert custom tags to help us find the position of A, B, and the pronoun after tokenization."""

        to_be_inserted = sorted([
            (row["A-offset"], " [[A-S]] ", len(row["A"]), " [[A-E]] "),
            (row["B-offset"], " [[B-S]] ", len(row["B"]), " [[B-E]] "),
            (row["Pronoun-offset"], " [[P-S]] ", len(row["Pronoun"]), " [[P-E]] ")
        ], key=lambda x: x[0], reverse=True)
        text = row["Text"]
        for offset, tag, l, etag in to_be_inserted:
            if text[offset+l:offset+l+2] == "'s":
                l += 2
            text = text[:offset] + tag + \
                    text[offset:offset+l] + etag + \
                    text[offset+l:]
        return text

    def extract_target(self,df):
        df["Neither"] = 0
        df.loc[~(df['A-coref'] | df['B-coref']), "Neither"] = 1
        df["target"] = 0
        df.loc[df['B-coref'] == 1, "target"] = 1
        df.loc[df["Neither"] == 1, "target"] = 2
        print(df.target.value_counts())
        return df

def ut_gap_df():
    gapdf = GAPdf()

if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    ut_gap_df()
    print('success!')
