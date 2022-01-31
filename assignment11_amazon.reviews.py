#!/usr/bin/env python
# coding: utf-8

# In[1]: # Pratik Ahire


# Import Libraries
import numpy as np 
import pandas as pd 
import string 
import spacy 

from matplotlib.pyplot import imread
from matplotlib import pyplot as plt
from wordcloud import WordCloud
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Load data sets
reviews = pd.read_csv("C:\\Users\\Admin\\Downloads\\AMZReviews_Product_B09G9JJT7M_50_Reviews.csv")
reviews


# In[3]:


reviews.drop(['ID','Username','ProfileURL','Rate','RateText','Format','Title','Helpful','Date','Thumbnails','Images','Video','Verified','ReviewURL'],axis=1,inplace=True)


# In[4]:


reviews.rename(columns={'Content':'text'},inplace=True)
reviews


# In[9]:


reviews_cl=reviews.dropna()
reviews_cl


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


# In[10]:


reviews_cl['text'] = reviews_cl.text.apply(clean)
reviews_cl.text


# In[12]:


reviews_cl = [text.strip() for text in reviews_cl.text] # remove both the leading and the trailing characters
reviews_cl = [text for text in reviews_cl if text] # removes empty strings, because they are considered in Python as False
reviews_cl[0:10]


# In[42]:


# Joining the list into one string/text
reviews_text = ' '.join(reviews_cl)
len(reviews_text)


# In[43]:


print(reviews_text)


# In[44]:


# Tokenization
from nltk.tokenize import word_tokenize
reviews_tokens = word_tokenize(reviews_text)
print(reviews_tokens)


# In[45]:


len(reviews_tokens) 


# In[46]:


# Stopwords
# Remove Stopwords
import nltk
from nltk.corpus import stopwords
my_stop_words = stopwords.words('english')
my_stop_words.append('the')
no_stop_tokens = [word for word in reviews_tokens if not word in my_stop_words]
print(no_stop_tokens[0:100])


# In[47]:


len(no_stop_tokens) 


# In[48]:


# Noramalize the data
lower_words = [text.lower() for text in no_stop_tokens]
print(lower_words[0:50])


# In[49]:


# NLP english language model of spacy library
nlp = spacy.load("en_core_web_sm")


# In[50]:


# lemmas being one of them, but mostly POS, which will follow later
doc = nlp(' '.join(lower_words))
print(doc[0:40])


# In[51]:


lemmas = [token.lemma_ for token in doc]
print(lemmas[0:40])


# In[52]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
tweetscv=cv.fit_transform(lemmas)


# In[53]:


print(cv.get_feature_names()[100:200])


# In[54]:


print(tweetscv.toarray()[100:200])


# In[55]:


print(tweetscv.toarray().shape) 


# In[56]:


cv_ngram_range=CountVectorizer(analyzer='word',ngram_range=(1,3),max_features=100)
bow_matrix_ngram=cv_ngram_range.fit_transform(lemmas)


# In[57]:


print(cv_ngram_range.get_feature_names())
print(bow_matrix_ngram.toarray())
# # 3. TF-IDF Vectorizer


# In[58]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidfv_ngram_max_features=TfidfVectorizer(norm='l2',analyzer='word',ngram_range=(1,3),max_features=500)
tfidf_matix_ngram=tfidfv_ngram_max_features.fit_transform(lemmas)


# In[59]:


print(tfidfv_ngram_max_features.get_feature_names())
print(tfidf_matix_ngram.toarray())


# In[60]:


clean_reviews=' '.join(lemmas)
clean_reviews


# In[61]:


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
                   colormap='Set1',stopwords=STOPWORDS).generate(clean_reviews)
plot_cloud(wordcloud)


# # Named Entity Recognition (NER)


# In[62]:


##Part Of Speech Tagging
nlp = spacy.load("en_core_web_sm")

one_block = clean_reviews
doc_block = nlp(one_block)
spacy.displacy.render(doc_block, style='ent', jupyter=True)


# In[63]:


for token in doc_block[500:600]:
    print(token, token.pos_)


# In[64]:


# Filtering the nouns and verbs only
nouns_verbs=[token.text for token in doc_block if token.pos_ in ('NOUN','VERB')]
print(nouns_verbs[500:600])


# In[65]:


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


# In[66]:


# Visualizing results (Barchart for top 10 nouns + verbs)
wd_df[0:10].plot.bar(x='word',figsize=(12,8),title='Top 10 nouns and verbs');


# In[ ]:




