#!/usr/bin/env python3
# -*- coding: utf-8 -*-

!pip install snorkel
from google.colab import drive
drive.mount('/content/drive/')
import os
os.chdir('drive/My Drive/dataset')

import pandas as pd
import io
import re

from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling import LabelingFunction
from snorkel.labeling import MajorityLabelVoter
from snorkel.labeling import LabelModel
from snorkel.labeling import filter_unlabeled_dataframe
from snorkel.utils import probs_to_preds

DISPLAY_ALL_TEXT = False
pd.set_option("display.max_colwidth", 0 if DISPLAY_ALL_TEXT else 150)

df = pd.read_csv("yelp.csv")    # originally named 'yelp_test.csv', with a size of 38000

df_train1 = df[['tag', 'text']]
df_train = df[['text']]
Y_train = df.tag.values 

df_dev = df[['tag','text']].sample(500, random_state = 3) # from the training
Y_dev = df_dev.tag.values

ABSTAIN = -1
POS = 1
NEG = 0

#--------expression type LFs-------------

pre_neg = ["no ", "not ", "never ", "n't "]

def expression_lookup(x, pre_pos, expression):
    flag = ABSTAIN
    for e in expression:
        for neg in pre_neg:
            string = neg + ".{0,20}" + e
            if re.search(string, x.text, flags=re.I):
                return NEG
        for pos in pre_pos:
            string = pos + ".{0,20}" + e
            if re.search(string, x.text, flags=re.I):
                return POS
    return ABSTAIN


def make_expression_lf(name, pre_pos, expression):
    return LabelingFunction(
        name=name,
        f=expression_lookup,
        resources=dict(pre_pos=pre_pos, expression=expression),
    )


expression_nexttime = make_expression_lf(name="expression_nexttime", 
    pre_pos=["will ", "'ll ", "would ", "'d ", "can't wait to "], 
    expression=[" back", " next time", " again", " return", " anymore"])
expression_recommend = make_expression_lf(name="expression_recommend", 
    pre_pos=["highly ", "do ", "would ", "definitely ", "certainly ", "strongly ", "i ", "we "], 
    expression=[" recommend", " nominate"])

#--------keyword type LFs-------------

negative_for_neg = ["in", "un", "im", "dis", "no ", "not ", "never ", "n't "]

def keyword_lookup(x, keywords_pos, keywords_neg):
    if keywords_pos:
        for key in keywords_pos:
            for neg in negative_for_neg:
                keywords_neg.append(neg+key)
    li = x.text.lower().split(" ")
    dic = {}
    for i in range(len(li)):
      dic[li[i]] = 1
    for neg in keywords_neg:
      if neg in dic:
        return NEG
    for pos in keywords_pos:
      if pos in dic:
        return POS
    return ABSTAIN


def make_keyword_lf(name, keywords_pos, keywords_neg):
    return LabelingFunction(
        name=name,
        f=keyword_lookup,
        resources=dict(keywords_pos=keywords_pos, keywords_neg=keywords_neg),
    )


keyword_general = make_keyword_lf(name="keyword_general", 
    keywords_pos=["outstanding", "perfect", "great", "good", "nice", "best", "excellent", "worthy", "awesome", "enjoy", "positive", "pleasant", "wonderful", "amazing"], 
    keywords_neg=["bad", "worst", "horrible", "awful", "terrible", "nasty", "shit", "distasteful", "dreadful", "negative"])
keyword_mood = make_keyword_lf(name="keyword_mood", 
    keywords_pos=["happy", "pleased", "delighted", "contented", "glad", "thankful", "satisfied"], 
    keywords_neg=["sad", "annoy", "disappointed", "frustrated", "upset", "irritated", "harassed", "angry", "pissed"])
keyword_service = make_keyword_lf(name="keyword_service", 
    keywords_pos=["friendly", "patient", "considerate", "enthusiastic", "attentive", "thoughtful", "kind", "caring", "helpful", "polite", "efficient", "prompt"], 
    keywords_neg=["slow", "offended", "rude", "indifferent", "arrogant"])
keyword_price = make_keyword_lf(name="keyword_price", 
    keywords_pos=["cheap", "reasonable", "inexpensive", "economical"], 
    keywords_neg=["overpriced", "expensive", "costly", "high-priced"])
keyword_environment = make_keyword_lf(name="keyword_environment", 
    keywords_pos=["clean", "neat", "quiet", "comfortable", "convenien", "tidy", "orderly", "cosy", "homely"], 
    keywords_neg=["noisy", "mess", "chaos", "dirty", "foul"])
keyword_food = make_keyword_lf(name="keyword_food", 
    keywords_pos=["tasty", "yummy", "delicious", "appetizing", "good-tasting", "delectable", "savoury", "luscious", "palatable"], 
    keywords_neg=["disgusting", "gross", "insipid"])
keyword_recommend = make_keyword_lf(name="keyword_recommend", 
    keywords_pos=["recommend"], 
    keywords_neg=[])

def keyword_price(x):
    keywords_pos=["cheap", "reasonable", "inexpensive", "economical"]
    keywords_neg=["overpriced", "expensive", "costly", "high-priced"]
    if any(word in x.text.lower() for word in keywords_neg):
        return NEG
    if any(word in x.text.lower() for word in keywords_pos):
        return POS
    return ABSTAIN

def keyword_environment(x):
    keywords_pos=["clean", "neat", "quiet", "comfortable", "convenient", "tidy", "orderly", "cosy", "homely"]
    keywords_neg=["noisy", "mess", "chaos", "dirty", "foul"]
    if any(word in x.text.lower() for word in keywords_neg):
        return NEG
    if any(word in x.text.lower() for word in keywords_pos):
        return POS
    return ABSTAIN

def keyword_service(x):
    keywords_pos=["friendly", "patient", "considerate", "attentive", "thoughtful", "helpful", "polite"]
    keywords_neg=["slow", "offended", "rude", "indifferent", "arrogant"]
    if any(word in x.text.lower() for word in keywords_neg):
        return NEG
    if any(word in x.text.lower() for word in keywords_pos):
        return POS
    return ABSTAIN






from snorkel.preprocess import preprocessor
from textblob import TextBlob

@preprocessor(memoize=True)
def textblob_sentiment(x):
    scores = TextBlob(x.text)
    x.polarity = scores.sentiment.polarity
    x.subjectivity = scores.sentiment.subjectivity
    return x

@labeling_function(pre=[textblob_sentiment])
def textblob_lf(x):
    if x.polarity < -0.5:
        return NEG
    if x.polarity > 0.5:
        return POS
    return ABSTAIN

lfs = [
    textblob_lf,
    keyword_recommend,
    keyword_general,
    keyword_mood,
    keyword_service,
    keyword_price,
    keyword_environment,
    keyword_food,
]

applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=df_train)    
L_dev = applier.apply(df=df_dev)
print("LF_analysis")
print(LFAnalysis(L=L_dev, lfs=lfs).lf_summary(Y=Y_dev))

df_train1["LF1"] = L_train[:,0]
df_train1["LF2"] = L_train[:,1]
df_train1["LF3"] = L_train[:,2]
df_train1["LF4"] = L_train[:,3]
df_train1["LF5"] = L_train[:,4]
df_train1["LF6"] = L_train[:,5]
df_train1["LF7"] = L_train[:,6]
df_train1["LF8"] = L_train[:,7]

df_train1.to_csv("yelp_LF.csv")

#--------train label model-------

label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train=L_train, n_epochs=100, lr=0.01, seed=123)
probs_train = label_model.predict_proba(L=L_train)
preds_train = probs_to_preds(probs=probs_train)
print(label_model.score(L=L_train, Y=Y_train))

df_result = df[['tag']]
df_result['pred'] = preds_train
df_result.to_csv("yelp_snorkel.csv")

df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
    X=df_train, y=probs_train, L=L_train
)
preds_train_filtered = probs_to_preds(probs=probs_train_filtered)
print(len(preds_train_filtered))

from google.colab import files
files.download('yelp_LF.csv')
files.download('yelp_snorkel.csv')