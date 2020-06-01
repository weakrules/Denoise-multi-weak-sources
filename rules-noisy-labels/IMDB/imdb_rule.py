#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import io
import re

from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling import LabelingFunction
from snorkel.labeling import MajorityLabelVoter

DISPLAY_ALL_TEXT = False
pd.set_option("display.max_colwidth", 0 if DISPLAY_ALL_TEXT else 150)

df = pd.read_csv("imdb.csv")
#--------test set-------------
df_test = df.iloc[24000:][['text']]
Y_test = df.iloc[24000:].tag.values
#--------train set------------
df = df.iloc[0:5000]  
df_train = df[['text']]
#--------validation set-------
df_dev = df[['text','tag']].sample(500, random_state = 3) # from the training
Y_dev = df_dev.tag.values

ABSTAIN = -1
POS = 1
NEG = 0

#--------expression type LFs-------------

pre_neg = ["no ", "not ", "never ", "n t ", "zero ", "0 ", "tiny ", "little ", "less", "rare", "worse"]

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
    pre_pos=["will ", " ll ", "would ", " d ", "can t wait to "], 
    expression=[" next time", " again", " rewatch", " anymore", " rewind"])
expression_recommend = make_expression_lf(name="expression_recommend", 
    pre_pos=["highly ", "do ", "would ", "definitely ", "certainly ", "strongly ", "i ", "we "], 
    expression=[" recommend", " nominate"])
expression_value = make_expression_lf(name="expression_value", 
    pre_pos=["high ", "timeless ", "priceless ", "has ", "great ", "of ", "real ", "instructive "], 
    expression=[" value", " quality", " meaning", " significance"])

#--------keyword type LFs-------------

negative_for_neg = ["in", "un", "im", "dis", "no ", "not ", "never ", "n t "]

def keyword_lookup(x, keywords_pos, keywords_neg):
    if keywords_pos:
        for key in keywords_pos:
            for neg in negative_for_neg:
                keywords_neg.append(neg+key)

    if any(word in x.text.lower() for word in keywords_neg):
        return NEG
    elif keywords_pos and any(word in x.text.lower() for word in keywords_pos):
        return POS
    return ABSTAIN


def make_keyword_lf(name, keywords_pos, keywords_neg):
    return LabelingFunction(
        name=name,
        f=keyword_lookup,
        resources=dict(keywords_pos=keywords_pos, keywords_neg=keywords_neg),
    )


keyword_general = make_keyword_lf(name="keyword_general", 
    keywords_pos=["masterpiece", "outstanding", "perfect", "great", "good", "nice", "best", "excellent", "worthy", "awesome", "enjoy", "positive", "pleasant", "wonderful", "amazing", "superb", "fantastic", "marvellous", "fabulous"], 
    keywords_neg=["bad", "worst", "horrible", "awful", "terrible", "crap", "shit", "garbage", "rubbish", "waste"])
keyword_actor = make_keyword_lf(name="keyword_actor", 
    keywords_pos=["beautiful", "handsome", "talented"], 
    keywords_neg=[])
keyword_finish = make_keyword_lf(name="keyword_finish", 
    keywords_pos=[], 
    keywords_neg=["fast forward", "n t finish"])
keyword_plot = make_keyword_lf(name="keyword_plot", 
    keywords_pos=["well written", "absorbing", "attractive", "innovative", "instructive", "interesting", "touching", "moving"], 
    keywords_neg=["to sleep", "fell asleep", "boring", "dull", "plain"])
keyword_compare = make_keyword_lf(name="keyword_compare", 
    keywords_pos=[], 
    keywords_neg=[" than this", " than the film", " than the movie"])

lfs = [
    expression_nexttime,
    expression_recommend,
    expression_value,
    keyword_compare,
    keyword_general,
    keyword_actor,
    keyword_finish,
    keyword_plot
]

applier = PandasLFApplier(lfs=lfs)
#L_train = applier.apply(df=df_train)    
L_dev = applier.apply(df=df_dev)
print("LF_analysis")
print(LFAnalysis(L=L_dev, lfs=lfs).lf_summary(Y=Y_dev))