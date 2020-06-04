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
