#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import pickle
from tensorflow.keras.layers import TextVectorization
import tensorflow
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings 
warnings.filterwarnings('ignore') 

vect_pickled = pickle.load(open(r"C:\Users\Renu\Downloads\P316\models\vect.pkl", "rb"))
model = pickle.load(open('models\model.pkl','rb'))


# In[6]:


vectorizer = TextVectorization.from_config(vect_pickled['config'])
vectorizer.set_weights(vect_pickled['weights'])


# In[7]:


import nltk
import spacy # language models
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()
nltk.download('punkt')
nltk.download('stopwords')
nlp = spacy.load('en_core_web_sm')

my_stop_words = stopwords.words('english')

print(len(my_stop_words))
my_stop_words.extend(['make','see','people','tell','see','give','please','take','would','think','try','good','life',
                      'really','use','want','need','one','article','like','stop','page','time','delete','edit','wikipedia',
                      'even','thing','come','way','man','make','go','get','well','see','know','wiki','org','little',
                      'stop','stephen','hawk','people','u','real','tme','call','right','need','wikize',
                      'nothing','name','leave','admin','back','never','day','find','world','read','say',
                      'talk','keep','put','fact','work','user','change','post','anything','let','world','big',
                      'hope','head','face','ur','ever','talk','piece','mean','comment','write','also','day',
                      'en','love','much','house','watch','look','much','let', 'person','thank','every','could','show','p','source','oh','new','add','still','care','editor',
                      'problem','source','information','friend','something','since','actully','many','someone','site',
                      'long','history','live','reason','place','another','word','show','around','point','revert','around',
                      'family','shall','wish','continue','maybe','must','else','eat','understand','account','boy', 'seriously','sure',
                      'remove','actually','message','guy','first','seem','anyone','guess','believe','www', 'ya', 'yea', 'yeah', 'year', 'year ago', 'year old',
                      'yes', 'yet', 'yo', 'york', 'young', 'youtube', 'zero','wikipedian', 'wikipedians','username', 'userpage', 'usual', 'utc', 'utter', 'utterly','uk', 'um', 'un','ng','mr',])

print(len(my_stop_words))


# In[51]:


def creat_cloud(text):
    
    wc = WordCloud(background_color="black",colormap="Oranges_r",
                   max_words=500,stopwords=my_stop_words, max_font_size= 60)
    wc.generate(text)
    plt.figure(figsize=(6,3),facecolor = None)
    plt.title("Word cloud", fontsize=10)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis('off')
    plt.savefig('static\images\cloud.png',bbox_inches='tight',pad_inches = 0.12)
    


# In[52]:


def word_distribution(text):
    text_token = word_tokenize(text)
    if len(text_token)>1000:
        rand_word = random.choices(text_token, k=1500)
    else:
        rand_word = text_token
    toxicity = list()
    for word in rand_word:
        word_vect = vectorizer(word)
        yhat = model.predict(np.expand_dims(word_vect, axis=0))[0].tolist()
        toxicity.append(yhat)
    df = pd.DataFrame(toxicity)
    a = np.round(((df.sum()/len(rand_word))*100).values, 2)
    plt.figure(figsize=(6, 3))
    ax = sns.barplot(x=['toxic','severe_toxic','obscene','threat','insult','identity_hate'],
                     y=a, palette='viridis')
    plt.ylabel('% of bad words in the text')
    plt.title('no. of Label Occurrences in the text')
    ax.bar_label(ax.containers[0], fontsize = 8)
    plt.yticks(ticks=[])
    plt.xlim(-0.5,5.5)
    plt.savefig('static\images\word_dist.png',bbox_inches='tight',pad_inches = 0.05)
    # Show the plot
    # make overall prediction
    input_text = vectorizer(text)
    prediction = model.predict(np.expand_dims(input_text, axis=0))
    pred_idx = np.nonzero(prediction)[1]
    toxic_level = ['toxic','sever_toxic','obscene','threat','insult','identity_hate']
    result = list(toxic_level[i] for i in pred_idx )
    if len(result)==0:
        statement = "The text content is mostly clean and non-toxic".split(' ')
        return ' '.join(statement)
    else:
        toxic_level = ', '.join(result[:-1])
        statement = ("The text includes words identified as "+toxic_level+" and " + result[-1]).split(' ')
        return ' '.join(statement)


# In[ ]:




