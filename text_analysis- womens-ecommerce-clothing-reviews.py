#!/usr/bin/env python
# coding: utf-8

# # We have Women's E-Commerce Clothing Reviews
# 
# # Let's Explore,  Analyse  and Visualize this Text Data 
# 
# 

# Data from https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews

# This is a Women’s Clothing E-Commerce dataset revolving around the reviews written by customers. Its nine supportive features offer a great environment to parse out the text through its multiple dimensions. Because this is real commercial data, it has been anonymized, and references to the company in the review text and body have been replaced with “retailer”.
# 
# ## Content
# This dataset includes 23486 rows and 10 feature variables. Each row corresponds to a customer review, and includes the variables:
# 
# - ###### Clothing ID:  
# Integer Categorical variable that refers to the specific piece being reviewed.
# - ###### Age:  
# Positive Integer variable of the reviewers age.
# - ###### Title: 
# String variable for the title of the review.
# - ###### Review Text: 
# String variable for the review body.
# - ###### Rating: 
# Positive Ordinal Integer variable for the product score granted by the customer from 1 Worst, to 5 Best.
# - ###### Recommended IND: 
# Binary variable stating where the customer recommends the product where 1 is recommended, 0 is not recommended.
# - ###### Positive Feedback Count: 
# Positive Integer documenting the number of other customers who found this review positive.
# - ###### Division Name: 
# Categorical name of the product high level division.
# - ###### Department Name: 
# Categorical name of the product department name.
# - ###### Class Name: 
# Categorical name of the product class name.
# 
# 
# 
# ## Acknowledgements 
# Anonymous but real source
# 
# 

# ## Reading for the project:
# - Text mining: https://en.wikipedia.org/wiki/Text_mining
# - explore: https://plot.ly/python/, 
# - visualization: https://bokeh.pydata.org/en/latest/
# - N-Gram: https://en.wikipedia.org/wiki/N-gram
# - scikit-learn’s CountVectorizer function: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
# - Part-Of-Speech Tagging (POS): https://en.wikipedia.org/wiki/Part-of-speech_tagging
# - TextBlob API: https://textblob.readthedocs.io/en/dev/
# - scattertext: https://github.com/JasonKessler/scattertext#using-scattertext-as-a-text-analysis-library-finding-characteristic-terms-and-their-associations)
# - spaCy libraries: https://github.com/explosion/spaCy
# 
# 
# 
# 

# In[1]:


import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings 
warnings.filterwarnings('ignore')
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (10.0, 6.0)
import plotly.graph_objs as go
import plotly.plotly as py
import cufflinks
pd.options.display.max_columns = 30
from IPython.core.interactiveshell import InteractiveShell
import plotly.figure_factory as ff
InteractiveShell.ast_node_interactivity = 'all'
from plotly.offline import iplot
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook
output_notebook()
from collections import Counter
import scattertext as st
import spacy
from pprint import pprint
import en_core_web_sm
nlp = spacy.load('en_core_web_sm')
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


# In[2]:


# the data
df = pd.read_csv('Womens Clothing E-Commerce Reviews.csv')
df.head()


# After a brief inspection of the data, we found there are a series of data pre-processing we have to conduct:
# 
# -Remove the “Title” feature.
# 
# -Remove the rows where “Review Text” were missing.
# 
# -Clean “Review Text” column.
# 
# -Using TextBlob to calculate sentiment polarity which lies in the range of (-1,1) where 1 means positive sentiment and -1 means a negative sentiment.
# 
# -Create new feature for the length of the review.
# 
# -Create new feature for the word count of the review.

# In[3]:


df.drop('Unnamed: 0', axis=1, inplace=True) #Remove the “Unnamed” column.
df.drop('Title', axis=1, inplace=True) #Remove the “Title” feature.
df = df[~df['Review Text'].isnull()]  #Remove the rows where “Review Text” were missing.

def preprocess(ReviewText):
    ReviewText = ReviewText.str.replace("(<br/>)", "")
    ReviewText = ReviewText.str.replace('(<a).*(>).*(</a>)', '')
    ReviewText = ReviewText.str.replace('(&amp)', '')
    ReviewText = ReviewText.str.replace('(&gt)', '')
    ReviewText = ReviewText.str.replace('(&lt)', '')
    ReviewText = ReviewText.str.replace('(\xa0)', ' ')  
    return ReviewText
df['Review Text'] = preprocess(df['Review Text']) #Clean “Review Text” column

df['polarity'] = df['Review Text'].map(lambda text: TextBlob(text).sentiment.polarity) #1 means positive review and -1 means negative review
df['review_len'] = df['Review Text'].astype(str).apply(len)  #Create new feature for the length of the review.
df['word_count'] = df['Review Text'].apply(lambda x: len(str(x).split())) #Create new feature for the word count of the review.


# To preview whether the sentiment polarity score works, we randomly select 5 positive review (polarity score=1):

# In[4]:


print('5 random positive reviews: \n')
cl = df.loc[df.polarity == 1, ['Review Text']].sample(5).values
for c in cl:
    print(c[0])


# Then randomly select 5 neutral reviews with the polarity score (zero):

# In[5]:


print('5 random neutral reviews: \n')
cl = df.loc[df.polarity == 0, ['Review Text']].sample(5).values
for c in cl:
    print(c[0])


# In[6]:


df.polarity.min() # most negative review


# In[7]:


df.loc[df.polarity < 0]


# In[8]:


df.loc[df.polarity == -0.9750000000000001]


# There were 1322 negative reviews and only 2 reviews with the most negative sentiment polarity score:

# In[9]:


print('Two  most negative reviews: \n')
cl = df.loc[df.polarity == -0.9750000000000001, ['Review Text']].sample(2).values
for c in cl:
    print(c[0])


# # Univariate visualization with Plotly

# Single-variable or univariate visualization is the simplest type of visualization which consists of observations on only a single characteristic or attribute. Univariate visualization includes histogram, bar plots and line charts.
# 
# ## Which reviews we have the most? positive or negative?
# 

# In[10]:


df['polarity'].iplot(
    kind='hist',
    bins=50,
    xTitle='polarity',
    linecolor='black',
    yTitle='count',
    title='Sentiment Polarity Distribution')


# Vast majority of the polarity are greater than 0, means most of them are positive.
# 
# ## How many  positive and negative reviews we have?

# In[11]:


#score granted by the customer from 1 Worst, to 5 Best
df['Rating'].iplot(
    kind='hist',
    xTitle='rating',
    linecolor='black',
    yTitle='count',
    title='Review Rating Distribution')


# The ratings are in align with the polarity score, that is, most of the ratings are pretty high at 4 or 5 ranges.
# 
# ## How old our reviewers?

# In[12]:


df['Age'].iplot(
    kind='hist',
    bins=100,
    xTitle='age',
    linecolor='black',
    yTitle='count',
    title='Reviewers Age Distribution')


# Most reviewers are in their 30s to 50s.
# 
# ## How long the review text ?

# In[13]:


df['review_len'].iplot(
    kind='hist',
    bins=30,
    xTitle='review length',
    linecolor='black',
    yTitle='count',
    title='Review Text Length Distribution')


# Most length of the text is more than 500

# ## How many words do we have in reviews? 

# In[14]:


df['word_count'].iplot(
     kind='hist',
     bins=100,
     xTitle='word count',
     linecolor='black',
     yTitle='count',
     title='Review Text Word Count Distribution') 


# Many people like to leave long reviews. 
# 
# For categorical features, we use bar chart to present the frequency.
# 
# 
# ## How many divisions we have? Which devision has the most reviews?

# In[15]:


df.groupby('Division Name').count()['Clothing ID'].iplot(kind='bar', yTitle='Count', linecolor='black', opacity=0.8,
                                                           title='Bar chart of Division Names', xTitle='Division Name')


# Store has three devision names. General division has the most number of reviews, and Initmates division has the least number of reviews.

# ## How many departments we have? Which departments has the most reviews?

# In[16]:


df.groupby('Department Name').count()['Clothing ID'].sort_values(ascending=False).iplot(kind='bar', 
                                                                                       yTitle='Count', 
                                                                                       linecolor='black', 
                                                                                       opacity=0.8,
                                                                                       title='Bar chart of Department Names', 
                                                                                       xTitle='Department Name')


# Tops department has the most reviews and Trend department has the least number of reviews.

# ## How many classes we have? Which class has the most reviews?

# In[17]:


df.groupby('Class Name').count()['Clothing ID'].sort_values(ascending=False).iplot(kind='bar', 
                                                                                   yTitle='Count', 
                                                                                   linecolor='black', 
                                                                                   opacity=0.8,
                                                                                   title='Bar chart of Class Names', xTitle='Class Name')


# we have 20 classes. Dresses, Knits and Blouses have the most reviews. 
# 
# 
# ## Relationship between  age groups and Rating

# In[18]:


df[['Rating', 'Age']].iplot(secondary_y='Age', secondary_y_title='Age',
    kind='box', yTitle='Rating', title='Box Plot of Age and Rating')


# Now we come to “Review Text” feature, before exploring this feature, we need to extract N-Gram features. N-grams are used to describe the number of words used as observation points, e.g., unigram means singly-worded, bigram means 2-worded phrase, and trigram means 3-worded phrase. In order to do this, we use scikit-learn’s CountVectorizer function.
# Lean more about 
# - N-Gram here: https://en.wikipedia.org/wiki/N-gram
# - scikit-learn’s CountVectorizer function here: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
# 
# First, it would be interesting to compare unigrams before and after removing stop words.
# 
# ## The distribution of top unigrams before removing stop words
# 

# In[19]:


def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(df['Review Text'], 20)
for word, freq in common_words:
    print(word, freq)
df1 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])


# In[20]:


df1.groupby('ReviewText').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 words in review before removing stop words')


# ## The distribution of top unigrams after removing stop words

# In[21]:


def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(df['Review Text'], 20)
for word, freq in common_words:
    print(word, freq)
df2 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])


# In[22]:


df2.groupby('ReviewText').sum()['count'].sort_values(ascending=False).iplot(
     kind='bar', yTitle='Count', linecolor='black', title='Top 20 words in review after removing stop words')


# Second, we want to compare bigrams before and after removing stop words.
# 
# ## The distribution of top bigrams before removing stop words

# In[23]:


def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_bigram(df['Review Text'], 20)
for word, freq in common_words:
    print(word, freq)
df3 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])


# In[24]:


df3.groupby('ReviewText').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 bigrams in review before removing stop words')


# ## Top bigrams after removing stop words

# In[25]:


def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_bigram(df['Review Text'], 20)
for word, freq in common_words:
    print(word, freq)
df4 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])


# In[26]:


df4.groupby('ReviewText').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 bigrams in review after removing stop words')


# Last, we compare trigrams before and after removing stop words.
# 
# ## The distribution of Top trigrams before removing stop words

# In[27]:


def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_trigram(df['Review Text'], 20)
for word, freq in common_words:
    print(word, freq)
df5 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])


# In[28]:


df5.groupby('ReviewText').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 trigrams in review before removing stop words')


# ## Top trigrams after removing stop words

# In[29]:


def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_trigram(df['Review Text'], 20)
for word, freq in common_words:
    print(word, freq)
df6 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])


# In[30]:


df6.groupby('ReviewText').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 trigrams in review after removing stop words')


# Part-Of-Speech Tagging (POS) is a process of assigning parts of speech to each word, such as noun, verb, adjective, etc. 
# 
# Learn more: https://en.wikipedia.org/wiki/Part-of-speech_tagging
# 
# 
# We use a simple TextBlob API to dive into POS of our “Review Text” feature in our data set, and visualize these tags.
# 
# Learn more: https://textblob.readthedocs.io/en/dev/
# 
# 
# 
# ## The distribution of top part-of-speech tags of review corpus

# In[31]:


#for this part we import nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')


# In[32]:


blob = TextBlob(str(df['Review Text']))
pos_df = pd.DataFrame(blob.tags, columns = ['word' , 'pos'])
pos_df = pos_df.pos.value_counts()[:20]
pos_df.iplot(
    kind='bar',
    xTitle='POS',
    yTitle='count', 
    title='Top 20 Part-of-speech tagging for review corpus')


# ## Distribution of word count by recommendations

# In[33]:


x1 = df.loc[df['Recommended IND'] == 1, 'word_count']
x0 = df.loc[df['Recommended IND'] == 0, 'word_count']

trace1 = go.Histogram(
    x=x0, name='Not recommended',
    opacity=0.75
)
trace2 = go.Histogram(
    x=x1, name = 'Recommended',
    opacity=0.75
)

data = [trace1, trace2]
layout = go.Layout(barmode = 'group', title='Distribution of Word Count Based on Recommendation')
fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='grouped histogram')


# Recommended reviews tend to be more lengthier than those not recommended reviews.

# In[34]:


y0 = df.loc[df['Division Name'] == 'General']['polarity']
y1 = df.loc[df['Division Name'] == 'General Petite']['polarity']
y2 = df.loc[df['Division Name'] == 'Initmates']['polarity']

trace0 = go.Box(
    y=y0,
    name = 'General',
    marker = dict(
        color = 'rgb(214, 12, 140)',
    )
)
trace1 = go.Box(
    y=y1,
    name = 'General Petite',
    marker = dict(
        color = 'rgb(0, 128, 128)',
    )
)
trace2 = go.Box(
    y=y2,
    name = 'Initmates',
    marker = dict(
        color = 'rgb(10, 140, 208)',
    )
)
data = [trace0, trace1, trace2]
layout = go.Layout(
    title = "Sentiment Polarity Boxplot of Division Name"
)

fig = go.Figure(data=data,layout=layout)
iplot(fig, filename = "Sentiment Polarity Boxplot of Division Name")


# The highest sentiment polarity score was achieved by all of three divisions, and the lowest sentiment polarity score was collected by General division.
# 
# We don't see any significant difference in terms of sentiment polarity between division names.

# Box plot is used to compare the sentiment polarity score, rating, review text lengths of each department or division of the e-commerce store.
# 
# ## What do the departments tell about Sentiment polarity

# In[35]:


y0 = df.loc[df['Department Name'] == 'Tops']['polarity']
y1 = df.loc[df['Department Name'] == 'Dresses']['polarity']
y2 = df.loc[df['Department Name'] == 'Bottoms']['polarity']
y3 = df.loc[df['Department Name'] == 'Intimate']['polarity']
y4 = df.loc[df['Department Name'] == 'Jackets']['polarity']
y5 = df.loc[df['Department Name'] == 'Trend']['polarity']

trace0 = go.Box(
    y=y0,
    name = 'Tops',
    marker = dict(
        color = 'rgb(214, 12, 140)',
    )
)
trace1 = go.Box(
    y=y1,
    name = 'Dresses',
    marker = dict(
        color = 'rgb(0, 128, 128)',
    )
)
trace2 = go.Box(
    y=y2,
    name = 'Bottoms',
    marker = dict(
        color = 'rgb(10, 140, 208)',
    )
)
trace3 = go.Box(
    y=y3,
    name = 'Intimate',
    marker = dict(
        color = 'rgb(12, 102, 14)',
    )
)
trace4 = go.Box(
    y=y4,
    name = 'Jackets',
    marker = dict(
        color = 'rgb(10, 0, 100)',
    )
)
trace5 = go.Box(
    y=y5,
    name = 'Trend',
    marker = dict(
        color = 'rgb(100, 0, 10)',
    )
)
data = [trace0, trace1, trace2, trace3, trace4, trace5]
layout = go.Layout(
    title = "Sentiment Polarity Boxplot of Department Name"
)

fig = go.Figure(data=data,layout=layout)
iplot(fig, filename = "Sentiment Polarity Boxplot of Department Name")


# The highest sentiment polarity score was achieved by all of the six departments except Trend department, and the lowest sentiment polarity score was collected by Tops department. And the Trend department has the lowest median polarity score. If you remember, the Trend department has the least number of reviews. This explains why it does not have as wide variety of score distribution as the other departments.
# 
# ## What do the departments tell about rating?
# 

# In[36]:


y0 = df.loc[df['Department Name'] == 'Tops']['Rating']
y1 = df.loc[df['Department Name'] == 'Dresses']['Rating']
y2 = df.loc[df['Department Name'] == 'Bottoms']['Rating']
y3 = df.loc[df['Department Name'] == 'Intimate']['Rating']
y4 = df.loc[df['Department Name'] == 'Jackets']['Rating']
y5 = df.loc[df['Department Name'] == 'Trend']['Rating']

trace0 = go.Box(
    y=y0,
    name = 'Tops',
    marker = dict(
        color = 'rgb(214, 12, 140)',
    )
)
trace1 = go.Box(
    y=y1,
    name = 'Dresses',
    marker = dict(
        color = 'rgb(0, 128, 128)',
    )
)
trace2 = go.Box(
    y=y2,
    name = 'Bottoms',
    marker = dict(
        color = 'rgb(10, 140, 208)',
    )
)
trace3 = go.Box(
    y=y3,
    name = 'Intimate',
    marker = dict(
        color = 'rgb(12, 102, 14)',
    )
)
trace4 = go.Box(
    y=y4,
    name = 'Jackets',
    marker = dict(
        color = 'rgb(10, 0, 100)',
    )
)
trace5 = go.Box(
    y=y5,
    name = 'Trend',
    marker = dict(
        color = 'rgb(100, 0, 10)',
    )
)
data = [trace0, trace1, trace2, trace3, trace4, trace5]
layout = go.Layout(
    title = "Rating Boxplot of Department Name"
)

fig = go.Figure(data=data,layout=layout)
iplot(fig, filename = "Rating Boxplot of Department Name")


# Except Trend department, all the other departments’ median rating were 5. Overall, the ratings are high and sentiment are positive in this review data set.
# 
# ## Review length by department

# In[37]:


y0 = df.loc[df['Department Name'] == 'Tops']['review_len']
y1 = df.loc[df['Department Name'] == 'Dresses']['review_len']
y2 = df.loc[df['Department Name'] == 'Bottoms']['review_len']
y3 = df.loc[df['Department Name'] == 'Intimate']['review_len']
y4 = df.loc[df['Department Name'] == 'Jackets']['review_len']
y5 = df.loc[df['Department Name'] == 'Trend']['review_len']

trace0 = go.Box(
    y=y0,
    name = 'Tops',
    marker = dict(
        color = 'rgb(214, 12, 140)',
    )
)
trace1 = go.Box(
    y=y1,
    name = 'Dresses',
    marker = dict(
        color = 'rgb(0, 128, 128)',
    )
)
trace2 = go.Box(
    y=y2,
    name = 'Bottoms',
    marker = dict(
        color = 'rgb(10, 140, 208)',
    )
)
trace3 = go.Box(
    y=y3,
    name = 'Intimate',
    marker = dict(
        color = 'rgb(12, 102, 14)',
    )
)
trace4 = go.Box(
    y=y4,
    name = 'Jackets',
    marker = dict(
        color = 'rgb(10, 0, 100)',
    )
)
trace5 = go.Box(
    y=y5,
    name = 'Trend',
    marker = dict(
        color = 'rgb(100, 0, 10)',
    )
)
data = [trace0, trace1, trace2, trace3, trace4, trace5]
layout = go.Layout(
    title = "Review length Boxplot of Department Name"
)

fig = go.Figure(data=data,layout=layout)
iplot(fig, filename = "Review Length Boxplot of Department Name")


# The median review length of Tops & Intimate departments are relative lower than those of the other departments.

# # Bivariate visualization with Plotly
# 
# Bivariate visualization is a type of visualization that consists two features at a time. It describes association or relationship between two features.
# 
# ## Distribution of sentiment polarity score by recommendations

# In[38]:


x1 = df.loc[df['Recommended IND'] == 1, 'polarity']
x0 = df.loc[df['Recommended IND'] == 0, 'polarity']

trace1 = go.Histogram(
    x=x0, name='Not recommended',
    opacity=0.75
)
trace2 = go.Histogram(
    x=x1, name = 'Recommended',
    opacity=0.75
)

data = [trace1, trace2]
layout = go.Layout(barmode='overlay', title='Distribution of Sentiment polarity of reviews based on Recommendation')
fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='overlaid histogram')


# It is obvious that reviews have higher polarity score are more likely to be recommended.
# 
# ## Distribution of ratings by recommendations

# In[39]:


x1 = df.loc[df['Recommended IND'] == 1, 'Rating']
x0 = df.loc[df['Recommended IND'] == 0, 'Rating']

trace1 = go.Histogram(
    x=x0, name='Not recommended',
    opacity=0.75
)
trace2 = go.Histogram(
    x=x1, name = 'Recommended',
    opacity=0.75
)

data = [trace1, trace2]
layout = go.Layout(barmode='overlay', title='Distribution of Sentiment polarity of reviews based on Recommendation')
fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='overlaid histogram')


# Recommended reviews have higher ratings than those of not recommended ones.
# 
# ## Distribution of review lengths by recommendations

# In[40]:


x1 = df.loc[df['Recommended IND'] == 1, 'review_len']
x0 = df.loc[df['Recommended IND'] == 0, 'review_len']

trace1 = go.Histogram(
    x=x0, name='Not recommended',
    opacity=0.75
)
trace2 = go.Histogram(
    x=x1, name = 'Recommended',
    opacity=0.75
)

data = [trace1, trace2]
layout = go.Layout(barmode = 'group', title='Distribution of Review Lengths Based on Recommendation')
fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='stacked histogram')


# Recommended reviews tend to be lengthier than those of not recommended reviews.

# ## 2D Density jointplot of sentiment polarity vs. rating

# In[41]:


trace1 = go.Scatter(
    x=df['polarity'], y=df['Rating'], mode='markers', name='points',
    marker=dict(color='rgb(102,0,0)', size=2, opacity=0.4)
)
trace2 = go.Histogram2dContour(
    x=df['polarity'], y=df['Rating'], name='density', ncontours=20,
    colorscale='Hot', reversescale=True, showscale=False
)
trace3 = go.Histogram(
    x=df['polarity'], name='Sentiment polarity density',
    marker=dict(color='rgb(102,0,0)'),
    yaxis='y2'
)
trace4 = go.Histogram(
    y=df['Rating'], name='Rating density', marker=dict(color='rgb(102,0,0)'),
    xaxis='x2'
)
data = [trace1, trace2, trace3, trace4]

layout = go.Layout(
    showlegend=False,
    autosize=False,
    width=600,
    height=550,
    xaxis=dict(
        domain=[0, 0.85],
        showgrid=False,
        zeroline=False
    ),
    yaxis=dict(
        domain=[0, 0.85],
        showgrid=False,
        zeroline=False
    ),
    margin=dict(
        t=50
    ),
    hovermode='closest',
    bargap=0,
    xaxis2=dict(
        domain=[0.85, 1],
        showgrid=False,
        zeroline=False
    ),
    yaxis2=dict(
        domain=[0.85, 1],
        showgrid=False,
        zeroline=False
    )
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='2dhistogram-2d-density-plot-subplots')


# ## 2D Density jointplot of age and sentiment polarity

# In[42]:


trace1 = go.Scatter(
    x=df['Age'], y=df['polarity'], mode='markers', name='points',
    marker=dict(color='rgb(102,0,0)', size=2, opacity=0.4)
)
trace2 = go.Histogram2dContour(
    x=df['Age'], y=df['polarity'], name='density', ncontours=20,
    colorscale='Hot', reversescale=True, showscale=False
)
trace3 = go.Histogram(
    x=df['Age'], name='Age density',
    marker=dict(color='rgb(102,0,0)'),
    yaxis='y2'
)
trace4 = go.Histogram(
    y=df['polarity'], name='Sentiment Polarity density', marker=dict(color='rgb(102,0,0)'),
    xaxis='x2'
)
data = [trace1, trace2, trace3, trace4]

layout = go.Layout(
    showlegend=False,
    autosize=False,
    width=600,
    height=550,
    xaxis=dict(
        domain=[0, 0.85],
        showgrid=False,
        zeroline=False
    ),
    yaxis=dict(
        domain=[0, 0.85],
        showgrid=False,
        zeroline=False
    ),
    margin=dict(
        t=50
    ),
    hovermode='closest',
    bargap=0,
    xaxis2=dict(
        domain=[0.85, 1],
        showgrid=False,
        zeroline=False
    ),
    yaxis2=dict(
        domain=[0.85, 1],
        showgrid=False,
        zeroline=False
    )
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='2dhistogram-2d-density-plot-subplots')


# There were few people are very positive or very negative. People who give neutral to positive reviews are more likely to be in their 30s. Probably people at these age are likely to be more active.

# # Modeling with LSA

# In[43]:


reindexed_data = df['Review Text']
tfidf_vectorizer = TfidfVectorizer(stop_words='english', use_idf=True, smooth_idf=True)
reindexed_data = reindexed_data.values
document_term_matrix = tfidf_vectorizer.fit_transform(reindexed_data)


# In[44]:


n_topics = 6
lsa_model = TruncatedSVD(n_components=n_topics)
lsa_topic_matrix = lsa_model.fit_transform(document_term_matrix)


# In[45]:


def get_keys(topic_matrix):
    '''
    returns an integer list of predicted topic 
    categories for a given topic matrix
    '''
    keys = topic_matrix.argmax(axis=1).tolist()
    return keys

def keys_to_counts(keys):
    '''
    returns a tuple of topic categories and their 
    accompanying magnitudes for a given list of keys
    '''
    count_pairs = Counter(keys).items()
    categories = [pair[0] for pair in count_pairs]
    counts = [pair[1] for pair in count_pairs]
    return (categories, counts)


# In[46]:


lsa_keys = get_keys(lsa_topic_matrix)
lsa_categories, lsa_counts = keys_to_counts(lsa_keys)


# In[47]:


def get_top_n_words(n, keys, document_term_matrix, tfidf_vectorizer):
    '''
    returns a list of n_topic strings, where each string contains the n most common 
    words in a predicted category, in order
    '''
    top_word_indices = []
    for topic in range(n_topics):
        temp_vector_sum = 0
        for i in range(len(keys)):
            if keys[i] == topic:
                temp_vector_sum += document_term_matrix[i]
        temp_vector_sum = temp_vector_sum.toarray()
        top_n_word_indices = np.flip(np.argsort(temp_vector_sum)[0][-n:],0)
        top_word_indices.append(top_n_word_indices)   
    top_words = []
    for topic in top_word_indices:
        topic_words = []
        for index in topic:
            temp_word_vector = np.zeros((1,document_term_matrix.shape[1]))
            temp_word_vector[:,index] = 1
            the_word = tfidf_vectorizer.inverse_transform(temp_word_vector)[0][0]
            topic_words.append(the_word.encode('ascii').decode('utf-8'))
        top_words.append(" ".join(topic_words))         
    return top_words


# In[48]:


top_n_words_lsa = get_top_n_words(3, lsa_keys, document_term_matrix, tfidf_vectorizer)

for i in range(len(top_n_words_lsa)):
    print("Topic {}: ".format(i+1), top_n_words_lsa[i])


# In[49]:


top_3_words = get_top_n_words(3, lsa_keys, document_term_matrix, tfidf_vectorizer)
labels = ['Topic {}: \n'.format(i) + top_3_words[i] for i in lsa_categories]

fig, ax = plt.subplots(figsize=(16,8))
ax.bar(lsa_categories, lsa_counts);
ax.set_xticks(lsa_categories);
ax.set_xticklabels(labels);
ax.set_ylabel('Number of review text');
ax.set_title('LSA topic counts');
plt.show();


# In[50]:


tsne_lsa_model = TSNE(n_components=2, perplexity=50, learning_rate=100, 
                        n_iter=2000, verbose=1, random_state=0, angle=0.75)
tsne_lsa_vectors = tsne_lsa_model.fit_transform(lsa_topic_matrix)


# In[51]:


def get_mean_topic_vectors(keys, two_dim_vectors):
    '''
    returns a list of centroid vectors from each predicted topic category
    '''
    mean_topic_vectors = []
    for t in range(n_topics):
        reviews_in_that_topic = []
        for i in range(len(keys)):
            if keys[i] == t:
                reviews_in_that_topic.append(two_dim_vectors[i])    
        
        reviews_in_that_topic = np.vstack(reviews_in_that_topic)
        mean_review_in_that_topic = np.mean(reviews_in_that_topic, axis=0)
        mean_topic_vectors.append(mean_review_in_that_topic)
    return mean_topic_vectors


# In[52]:


colormap = np.array([
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
    "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
    "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
    "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5" ])
colormap = colormap[:n_topics]


# In[53]:


top_3_words_lsa = get_top_n_words(3, lsa_keys, document_term_matrix, tfidf_vectorizer)
lsa_mean_topic_vectors = get_mean_topic_vectors(lsa_keys, tsne_lsa_vectors)

plot = figure(title="t-SNE Clustering of {} LSA Topics".format(n_topics), plot_width=700, plot_height=700)
plot.scatter(x=tsne_lsa_vectors[:,0], y=tsne_lsa_vectors[:,1], color=colormap[lsa_keys])

for t in range(n_topics):
    label = Label(x=lsa_mean_topic_vectors[t][0], y=lsa_mean_topic_vectors[t][1], 
                  text=top_3_words_lsa[t], text_color=colormap[t])
    plot.add_layout(label)
    
show(plot)


# # Finding characteristic terms and their associations
# Sometimes we want to analyzes words used by different categories and outputs some notable term associations. We will use scattertext (https://github.com/JasonKessler/scattertext#using-scattertext-as-a-text-analysis-library-finding-characteristic-terms-and-their-associations) and spaCy libraries (https://github.com/explosion/spaCy) to accomplish these.
# 
# First, we need to turn the data frame into a Scattertext Corpus. To look for differences in department name, set the category_colparameter to 'Department Names', and use the review present in the Review Text column, to analyze by setting the text col parameter. Finally, pass a spaCy model in to the nlp argument and call build() to construct the corpus.
# 
# Following are the terms that differentiate the review text from a general English corpus.

# In[54]:


corpus = st.CorpusFromPandas(df, category_col='Department Name', text_col='Review Text', nlp=nlp).build()
print(list(corpus.get_scaled_f_scores_vs_background().index[:10]))


# Following are the terms in review text that are most associated with the Tops department:

# In[55]:


term_freq_df = corpus.get_term_freq_df()
term_freq_df['Tops Score'] = corpus.get_scaled_f_scores('Tops')
pprint(list(term_freq_df.sort_values(by='Tops Score', ascending=False).index[:10]))


# Here are the terms that are most associated with Dresses department:

# In[56]:


term_freq_df['Dresses Score'] = corpus.get_scaled_f_scores('Dresses')
pprint(list(term_freq_df.sort_values(by='Dresses Score', ascending=False).index[:10]))


# ## Topic Modeling Review Text

# Finally, we want to explore topic modeling algorithm to this data set, to see whether it would provide any benefit, and fit with what we are doing for our review text feature.
# 
# We will experiment with Latent Semantic Analysis (LSA) technique in topic modeling.
# 
# - Generating our document-term matrix from review text to a matrix of TF-IDF features.(https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
# - LSA model replaces raw counts in the document-term matrix with a TF-IDF score.
# - Perform dimensionality reduction on the document-term matrix using truncated SVD.(https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html)
# - Because the number of department is 6, we set n_topics=6.
# - Taking the argmax of each review text in this topic matrix will give the predicted topics of each review text in the data. We can then sort these into counts of each topic.
# - To better understand each topic, we will find the most frequent three words in each topic.

# In[57]:


reindexed_data = df['Review Text']
tfidf_vectorizer = TfidfVectorizer(stop_words='english', use_idf=True, smooth_idf=True)
reindexed_data = reindexed_data.values
document_term_matrix = tfidf_vectorizer.fit_transform(reindexed_data)
n_topics = 6
lsa_model = TruncatedSVD(n_components=n_topics)
lsa_topic_matrix = lsa_model.fit_transform(document_term_matrix)

def get_keys(topic_matrix):
    '''
    returns an integer list of predicted topic 
    categories for a given topic matrix
    '''
    keys = topic_matrix.argmax(axis=1).tolist()
    return keys

def keys_to_counts(keys):
    '''
    returns a tuple of topic categories and their 
    accompanying magnitudes for a given list of keys
    '''
    count_pairs = Counter(keys).items()
    categories = [pair[0] for pair in count_pairs]
    counts = [pair[1] for pair in count_pairs]
    return (categories, counts)
    
lsa_keys = get_keys(lsa_topic_matrix)
lsa_categories, lsa_counts = keys_to_counts(lsa_keys)

def get_top_n_words(n, keys, document_term_matrix, tfidf_vectorizer):
    '''
    returns a list of n_topic strings, where each string contains the n most common 
    words in a predicted category, in order
    '''
    top_word_indices = []
    for topic in range(n_topics):
        temp_vector_sum = 0
        for i in range(len(keys)):
            if keys[i] == topic:
                temp_vector_sum += document_term_matrix[i]
        temp_vector_sum = temp_vector_sum.toarray()
        top_n_word_indices = np.flip(np.argsort(temp_vector_sum)[0][-n:],0)
        top_word_indices.append(top_n_word_indices)   
    top_words = []
    for topic in top_word_indices:
        topic_words = []
        for index in topic:
            temp_word_vector = np.zeros((1,document_term_matrix.shape[1]))
            temp_word_vector[:,index] = 1
            the_word = tfidf_vectorizer.inverse_transform(temp_word_vector)[0][0]
            topic_words.append(the_word.encode('ascii').decode('utf-8'))
        top_words.append(" ".join(topic_words))         
    return top_words
    
    top_n_words_lsa = get_top_n_words(3, lsa_keys, document_term_matrix, tfidf_vectorizer)

for i in range(len(top_n_words_lsa)):
    print("Topic {}: ".format(i+1), top_n_words_lsa[i])


# In[58]:


top_3_words = get_top_n_words(3, lsa_keys, document_term_matrix, tfidf_vectorizer)
labels = ['Topic {}: \n'.format(i) + top_3_words[i] for i in lsa_categories]
fig, ax = plt.subplots(figsize=(16,8))
ax.bar(lsa_categories, lsa_counts);
ax.set_xticks(lsa_categories);
ax.set_xticklabels(labels);
ax.set_ylabel('Number of review text');
ax.set_title('LSA topic counts');
plt.show();


# By looking at the most frequent words in each topic, we have a sense that we may not reach any degree of separation across the topic categories. In other words, we could not separate review text by departments using topic modeling techniques.
# 
# Topic modeling techniques have a number of important limitations. To begin, the term “topic” is somewhat ambigious, and by now it is perhaps clear that topic models will not produce highly nuanced classification of texts for our data.
# 
# In addition, we can observe that the vast majority of the review text are categorized to the first topic (Topic 0). The t-SNE visualization of LSA topic modeling won’t be pretty.

# The credit for teaching me phyton analisys goes to the Scientist, Susan Li from Toronto, Canada. Have a wonderful Day!
