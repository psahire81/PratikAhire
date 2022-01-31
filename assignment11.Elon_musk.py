#!/usr/bin/env python
# coding: utf-8

# In[1]: # Pratik Ahire


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import string # special operations on strings
import spacy # language models

from matplotlib.pyplot import imread
from matplotlib import pyplot as plt
from wordcloud import WordCloud
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import pandas
tweets=pd.read_csv("C:\\Users\\Admin\\Downloads\\Elon_musk.csv",error_bad_lines=False)


# In[3]:


tweets.head()


# In[4]:


tweets.rename(columns={'Text':'text'},inplace=True)


# In[5]:


tweets.head()


# In[6]:


import re #regular expression
import string
# Remove Punctuation

def clean_text(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub("[0-9" "]+"," ",text)
    text = re.sub('[‘’“”…]', '', text)
    return text

clean = lambda x: clean_text(x)


# In[7]:


tweets['text'] = tweets.text.apply(clean)
tweets.text


# In[8]:


tweets = [text.strip() for text in tweets.text] # remove both the leading and the trailing characters
tweets = [text for text in tweets if text] # removes empty strings, because they are considered in Python as False
tweets


# In[9]:


# Joining the list into one string/text
tweets_text = ' '.join(tweets)
len(tweets_text)  #130001


# In[10]:


print(tweets_text[0:500])


# In[11]:


# Tokenization
from nltk.tokenize import word_tokenize
tweets_tokens = word_tokenize(tweets_text)
print(tweets_tokens[0:500])


# In[12]:


len(tweets_tokens)    #19609


# In[13]:


# Stopwords
# Remove Stopwords
import nltk
from nltk.corpus import stopwords
my_stop_words = stopwords.words('english')
my_stop_words.append('the')
no_stop_tokens = [word for word in tweets_tokens if not word in my_stop_words]
print(no_stop_tokens[0:100])


# In[14]:


len(no_stop_tokens)     #13414


# In[15]:


# Noramalize the data
lower_words = [text.lower() for text in no_stop_tokens]
print(lower_words[0:50])


# In[16]:


# NLP english language model of spacy library
nlp = spacy.load("en_core_web_sm")


# In[17]:


# lemmas being one of them, but mostly POS, which will follow later
doc = nlp(' '.join(lower_words))
print(doc[0:40])


# In[18]:


lemmas = [token.lemma_ for token in doc]
print(lemmas[0:40])


# In[19]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
tweetscv=cv.fit_transform(lemmas)


# In[20]:


print(cv.get_feature_names()[100:200])


# In[21]:


print(tweetscv.toarray()[100:200])


# In[22]:


print(tweetscv.toarray().shape)   # (13419, 3958)


# # 2. CountVectorizer with N-grams (Bigrams & Trigrams)


# In[23]:


cv_ngram_range=CountVectorizer(analyzer='word',ngram_range=(1,3),max_features=100)
bow_matrix_ngram=cv_ngram_range.fit_transform(lemmas)


# In[24]:


print(cv_ngram_range.get_feature_names())
print(bow_matrix_ngram.toarray())


# In[25]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidfv_ngram_max_features=TfidfVectorizer(norm='l2',analyzer='word',ngram_range=(1,3),max_features=500)
tfidf_matix_ngram=tfidfv_ngram_max_features.fit_transform(lemmas)


# In[26]:


print(tfidfv_ngram_max_features.get_feature_names())
print(tfidf_matix_ngram.toarray())


# In[27]:


clean_tweets=' '.join(lemmas)
clean_tweets


# In[28]:


# Define a function to plot word cloud
def plot_cloud(wordcloud):
    plt.figure(figsize=(40,30))
    plt.imshow(wordcloud)
    plt.axis('off')
    
# Generate Word Cloud

from wordcloud import WordCloud, STOPWORDS

STOPWORDS.add('pron')
STOPWORDS.add('rt')
STOPWORDS.add('yeah')
wordcloud=WordCloud(width=3000,height=2000,background_color='black',max_words=50,
                   colormap='Set1',stopwords=STOPWORDS).generate(clean_tweets)
plot_cloud(wordcloud)


# # Named Entity Recognition (NER)


# In[29]:


##Part Of Speech Tagging
nlp = spacy.load("en_core_web_sm")

one_block = clean_tweets
doc_block = nlp(one_block)
spacy.displacy.render(doc_block, style='ent', jupyter=True)


# In[30]:


for token in doc_block[500:600]:
    print(token, token.pos_)


# In[31]:


# Filtering the nouns and verbs only
nouns_verbs=[token.text for token in doc_block if token.pos_ in ('NOUN','VERB')]
print(nouns_verbs[500:600])


# In[32]:


# Counting the noun & verb tokens
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

X=cv.fit_transform(nouns_verbs)
sum_words=X.sum(axis=0)

words_freq=[(word,sum_words[0,idx]) for word,idx in cv.vocabulary_.items()]
words_freq=sorted(words_freq, key=lambda x: x[1], reverse=True)

wd_df=pd.DataFrame(words_freq)
wd_df.columns=['word','count']
wd_df[0:10] # viewing top ten results


# In[33]:


# Visualizing results (Barchart for top 10 nouns + verbs)
wd_df[0:10].plot.bar(x='word',figsize=(12,8),title='Top 10 nouns and verbs');


# In[34]:


tweets = pd.read_csv("C:\\Users\\Admin\\Downloads\\Elon_musk.csv",encoding='Latin-1')
tweets.drop(['Unnamed: 0'],inplace=True,axis=1)
tweets = tweets.rename({'Text': 'text'}, axis=1)


# In[35]:


import re #regular expression
import string
# Remove Punctuation

def clean_text(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub("[0-9" "]+"," ",text)
    text = re.sub('[‘’“”…]', '', text)
    return text

clean = lambda x: clean_text(x)


# In[36]:


tweets['text'] = tweets.text.apply(clean)


# In[37]:


tweets = [text.strip() for text in tweets.text] 
tweets = [text for text in tweets if text] 
tweets


# In[38]:


from nltk import tokenize
sentences = tokenize.sent_tokenize(' '.join(tweets))
sentences


# In[39]:


sent_df = pd.DataFrame(tweets,columns=['sentence'])
sent_df


# In[40]:


affin = pd.read_csv('C:\\Users\\Admin\\OneDrive\\Desktop\\data science\\Afinn.csv.xls', sep=',', encoding='latin-1')


# In[41]:


affinity_scores=affin.set_index('word')['value'].to_dict()
affinity_scores


# In[42]:


# Custom function: score each word in a sentence in lemmatised form, but calculate the score for the whole original sentence
nlp=spacy.load('en_core_web_sm')
sentiment_lexicon=affinity_scores

def calculate_sentiment(text:str=None):
    sent_score=0
    if text:
        sentence=nlp(text)
        for word in sentence:
            sent_score+=sentiment_lexicon.get(word.lemma_,0)
    return sent_score


# In[43]:


# manual testing
calculate_sentiment(text='great')


# In[44]:


# Calculating sentiment value for each sentence
sent_df['sentiment_value']=sent_df['sentence'].apply(calculate_sentiment)
sent_df['sentiment_value']


# In[45]:


sent_df.sort_values(by='sentiment_value')


# In[46]:


# Sentiment score of the whole review
sent_df['sentiment_value'].describe()


# In[47]:


#count    1974.000000
#mean        0.738095
#std         1.822537
#min        -7.000000
#25%         0.000000
#50%         0.000000
#75%         2.000000
#max        12.000000
#Name: sentiment_value, dtype: float64


# In[48]:


# negative sentiment score of the whole review
sent_df[sent_df['sentiment_value']<=0]


# In[49]:


# positive sentiment score of the whole review
sent_df[sent_df['sentiment_value']>0]


# In[50]:


# Adding index cloumn
sent_df['index']=range(0,len(sent_df))
sent_df


# In[51]:


# Plotting the sentiment value for whole review
import seaborn as sns
plt.figure(figsize=(15,10))
sns.distplot(sent_df['sentiment_value'])


# In[52]:


# Plotting the line plot for sentiment value of whole review
plt.figure(figsize=(15,10))
sns.lineplot(y='sentiment_value',x='index',data=sent_df)


# In[ ]:




