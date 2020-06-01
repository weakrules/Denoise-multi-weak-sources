#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# !pip install snorkel
# from google.colab import drive
# drive.mount('/content/drive/')
# import os
# os.chdir('drive/My Drive/dataset')

import pandas as pd
import io
import re

import torch

from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling import LabelingFunction
from snorkel.labeling import MajorityLabelVoter
from snorkel.labeling import LabelModel
from snorkel.labeling import filter_unlabeled_dataframe
from snorkel.utils import probs_to_preds

from sklearn.feature_extraction.text import CountVectorizer

#--------expression type LFs-------------
number_keywords = 2
def keyword_lookup(x, keywords, label):
    # count = 0
    
    # for word in keywords:
    #     for i in x.text.lower().split():
    #         if(word == i):
    #             count += 1
    #             break
    
    # if(count >= number_keywords):
    #     return label
    # print(x)
    if any(word in x.text.lower() for word in keywords):
        return label
    else: 
        return ABSTAIN


def make_keyword_lf(keywords, label):
    return LabelingFunction(
        name=f"keyword_{keywords[0]}",
        f=keyword_lookup,
        resources=dict(keywords=keywords, label=label),
    )

def lf():
    #### 0: international
    r1 = ["atomic", "captives", "baghdad", "israeli", "iraqis", "iranian", "afghanistan", "wounding", "terrorism", "soldiers", \
    "palestinians", "palestinian", "policemen", "iraqi", "terrorist", 'north korea', 'korea', \
    'israel', 'u.n.', 'egypt', 'iran', 'iraq', 'nato', 'armed', 'peace']
    r2= [' war ', 'prime minister', 'president', 'commander', 'minister',  'annan', "military", "militant", "kill", 'operator']
    #### 1: sports
    r3 = ["goals", "bledsoe", "coaches",  "touchdowns", "kansas", "rankings", "no.", \
     "champ", "cricketers", "hockey", "champions", "quarterback", 'club', 'team',  'baseball', 'basketball', 'soccer', 'football', 'boxing',  'swimming', \
     'world cup', 'nba',"olympics","final", "finals", 'fifa',  'racist', 'racism'] 
    r4 = ['athlete',  'striker', 'defender', 'goalkeeper',  'midfielder', 'shooting guard', 'power forward', 'point guard', 'pitcher', 'catcher', 'first base', 'second base', 'third base','shortstop','fielder']
    #r3 = ["tech", "digital", "internet", "mobile"]
    r5=['lakers','chelsea', 'piston','cavaliers', 'rockets', 'clippers','ronaldo', \
        'celtics', 'hawks','76ers', 'raptors', 'pacers', 'suns', 'warriors','blazers','knicks','timberwolves', 'hornets', 'wizards', 'nuggets', 'mavericks', 'grizzlies', 'spurs', \
        'cowboys', 'redskins', 'falcons', 'panthers', 'eagles', 'saints', 'buccaneers', '49ers', 'cardinals', 'texans', 'seahawks', 'vikings', 'patriots', 'colts', 'jaguars', 'raiders', 'chargers', 'bengals', 'steelers', 'browns', \
        'braves','marlins','mets','phillies','cubs','brewers','cardinals', 'diamondbacks','rockies', 'dodgers', 'padres', 'orioles', 'sox', 'yankees', 'jays', 'sox', 'indians', 'tigers', 'royals', 'twins','astros', 'angels', 'athletics', 'mariners', 'rangers', \
        'arsenal', 'burnley', 'newcastle', 'leicester', 'manchester united', 'everton', 'southampton', 'hotspur','tottenham', 'fulham', 'watford', 'sheffield','crystal palace', 'derby', 'charlton', 'aston villa', 'blackburn', 'west ham', 'birmingham city', 'middlesbrough', \
        'real madrid', 'barcelona', 'villarreal', 'valencia', 'betis', 'espanyol','levante', 'sevilla', 'juventus', 'inter milan', 'ac milan', 'as roma', 'benfica', 'porto', 'getafe', 'bayern', 'schalke', 'bremen', 'lyon', 'paris saint', 'monaco', 'dynamo']
    #### 3:tech
    #r4 =  ["tech", "digital", "internet", "mobile","hardware", "computer", "telescope", "phone", "robot", "router", "processor", "software", "ios", "browser"] #"wireless","chip", "pc", 
    r6 = ["technology", "engineering", "science", "research", "cpu", "windows", "unix", "system", 'computing',  'compute']#, "wireless","chip", "pc", ]
    #r6 =  ["software", "ios", "os", "browser"]
    # r7 = ["google", "apple", "microsoft", "nasa", "amazon", "yahoo", "cisco", "intel", "dell", \
    # "oracle", "hp", "ibm", "siemens", "nokia","motorola", "samsung", "toshiba", "sony"]
    r7= ["google", "apple", "microsoft", "nasa", "yahoo", "intel", "dell", \
    'huawei',"ibm", "siemens", "nokia", "samsung", 'panasonic', \
     't-mobile', 'nvidia', 'adobe', 'salesforce', 'linkedin', 'silicon', 'wiki'
    ]
    #### 2:business
    # r8 = ["business", "stock", "market", "industr", "trade", "sale", "financ", "goods", "retail"]
    r8= ["stock", "account", "financ", "goods", "retail", 'economy', 'chairman', 'bank', 'deposit', 'economic', 'dow jones', 'index', '$',  'percent', 'interest rate', 'growth', 'profit', 'tax', 'loan',  'credit', 'invest']
    # r9 = ["delta", "cola", "fort", "toyota", "costco", "gucci", 'citibank']
    r9= ["delta", "cola", "toyota", "costco", "gucci", 'citibank', 'airlines']
    #r9 = ['$', 'reuters', 'percent']#, '$', 'percent']
    return r1,r2, r3,r4,r5, r6,r7, r8,r9

  

ABSTAIN = -1 # todo: change it for imployloss / wendi_proj
politics = 0
sports = 1
business = 2
technology = 3

r1,r2, r3,r4,r5, r6,r7, r8,r9 = lf()

keyword_r1 = make_keyword_lf(keywords=r1, label=politics)

keyword_r2 = make_keyword_lf(keywords=r2, label=politics)

keyword_r3 = make_keyword_lf(keywords=r3, label=sports)

keyword_r4 = make_keyword_lf(keywords=r4, label=sports)

keyword_r5 = make_keyword_lf(keywords=r5, label=sports)

keyword_r6 = make_keyword_lf(keywords=r6, label=technology)

keyword_r7 = make_keyword_lf(keywords=r7, label=technology)

keyword_r8 = make_keyword_lf(keywords=r8, label=business)

keyword_r9 = make_keyword_lf(keywords=r9, label=business)


#--------get data---------------

DISPLAY_ALL_TEXT = False
pd.set_option("display.max_colwidth", 0 if DISPLAY_ALL_TEXT else 150)

df = pd.read_csv("AGnews.csv")  # originally named 'AGnews_train.csv', with a size of 120000

df_train1 = df[['tag', 'text']]
df_train = df[['text']]
Y_train = df.tag.values 

df_dev = df[['tag','text']].sample(10000, random_state = 3) # from the training
Y_dev = df_dev.tag.values




# keywords_pt = torch.load('key_words.pt')
# textrank_keywords = torch.load('textrank_keywords.pt')

# key1 = textrank_keywords['agnews_pol']
# key2 = textrank_keywords['agnews_sp']
# key3 = textrank_keywords['agnews_bus']
# key4 = textrank_keywords['agnews_tech']

    
# keyword_politics = make_keyword_lf(keywords=keywords_pt['kw0'], label=politics)

# keyword_sports =  make_keyword_lf(keywords=keywords_pt['kw1'], label=sports)

# keyword_business_tfidf = make_keyword_lf(keywords=keywords_pt['kw2'], label=business)

# keyword_technology_tfidf = make_keyword_lf(keywords=keywords_pt['kw3'], label=technology)

# keyword_technology = make_keyword_lf(keywords=["tech", "digital", "internet", "mobile"], label=technology)

# keyword_hardware = make_keyword_lf(keywords=["hardware", "computer", "telescope", "phone", "robot", "router", "chip", "pc", "processor", "wireless"], label=technology)

# keyword_software = make_keyword_lf(keywords=["software", "ios", "os", "browser"], label=technology)

# keyword_tech_org = make_keyword_lf(keywords=["google", "apple", "microsoft", "nasa", "amazon", "yahoo", "cisco", "intel", "dell", "oracle", "hp", "ibm", "siemens", "nokia", "motorola", "samsung", "toshiba", "sony"], label=technology)

# keyword_business = make_keyword_lf(keywords=["business", "stock", "market", "industr", "trade", "sale", "financ", "goods", "retail"], label=business)

# keyword_busi_company = make_keyword_lf(keywords=["delta", "cola", "fort", "toyota", "costco", "gucci"], label=business)

# keyword_rank_pol = make_keyword_lf(keywords=key1, label=politics)

# keyword_rank_sp = make_keyword_lf(keywords=key2, label=sports)

# keyword_rank_bus = make_keyword_lf(keywords=key3, label=business)

# keyword_rank_tech = make_keyword_lf(keywords=key4, label=technology)


# lfs = [
#     keyword_politics,
#     keyword_sports,
#     keyword_business_tfidf,
#     keyword_technology_tfidf,
#     keyword_technology,
#     keyword_hardware,
#     keyword_software,
#     keyword_tech_org,
#     keyword_business,
#     keyword_busi_company]   
# #     keyword_rank_pol,
# #     keyword_rank_sp,
# #     keyword_rank_bus,
# #     keyword_rank_tech
# # ]

lfs = [keyword_r1,
        keyword_r2,
        keyword_r3,
        keyword_r4,
        keyword_r5,
        keyword_r6,
        keyword_r7,
        keyword_r8,
        keyword_r9]

applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=df_train)    
L_dev = applier.apply(df=df_dev)
print("LF_analysis")
print(LFAnalysis(L=L_dev, lfs=lfs).lf_summary(Y=Y_dev))

# from summa import keywords

# agnews_pol = ""
# agnews_sp = ""
# agnews_bus = ""
# agnews_tech = ""

# for i in range(10000):
#     if(df['tag'].astype(str)[i] == '1'):
#         agnews_pol += df['text'].astype(str)[i]
#     elif(df['tag'].astype(str)[i] == '2'):
#         agnews_sp += df['text'].astype(str)[i]
#     elif(df['tag'].astype(str)[i] == '3'):
#         agnews_bus += df['text'].astype(str)[i]
#     elif(df['tag'].astype(str)[i] == '4'):
#         agnews_tech += df['text'].astype(str)[i]

# key1 = keywords.keywords(agnews_pol, ratio=0.01, split=True)
# key2 = keywords.keywords(agnews_sp, ratio=0.01, split=True)
# key3 = keywords.keywords(agnews_bus, ratio=0.01, split=True)
# key4 = keywords.keywords(agnews_tech, ratio=0.01, split=True)

# textrank_keywords = {
#     'agnews_pol': key1,
#     'agnews_sp': key2,
#     'agnews_bus': key3,
#     'agnews_tech': key4
#     }

# (torch.save(textrank_keywords, 'textrank_keywords.pt'))

# ------------- save ---------------- 
df_train1["LF1"] = L_train[:,0]
df_train1["LF2"] = L_train[:,1]
df_train1["LF3"] = L_train[:,2]
df_train1["LF4"] = L_train[:,3]
df_train1["LF5"] = L_train[:,4]
df_train1["LF6"] = L_train[:,5]
df_train1["LF7"] = L_train[:,6]
df_train1["LF8"] = L_train[:,7]
df_train1["LF9"] = L_train[:,8]

df_train1.to_csv("agnews_LF.csv")

#--------train label model-------

# label_model = LabelModel(cardinality=5, verbose=True)
# label_model.fit(L_train=L_train, n_epochs=100, lr=0.01, seed=123)
# probs_train = label_model.predict_proba(L=L_train)
# preds_train = probs_to_preds(probs=probs_train)
# print(label_model.score(L=L_train, Y=Y_train))

# df_result = df[['tag']]
# df_result['pred'] = preds_train
# df_result.to_csv("agnews_snorkel.csv")

# df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
#     X=df_train, y=probs_train, L=L_train
# )
# preds_train_filtered = probs_to_preds(probs=probs_train_filtered)
# print(len(preds_train_filtered))

# from google.colab import files
# files.download('agnews_LF.csv')
# files.download('agnews_snorkel.csv')