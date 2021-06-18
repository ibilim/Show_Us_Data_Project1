# 1.st Approach:
# Extract all possible sentences that include data then run topic analysis
from nltk.corpus import stopwords
import gensim
from gensim import corpora, models
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
import re
from gensim.models.ldamodel import LdaModel
#load text in json format
f_hand=open('2100032a-7c33-4bff-97ef-690822c43466.json','r')
text=js.load(f_hand)
#convert json to string
full_text=''
for i in range(0,len(text)):
    full_text=full_text+' '+text[i]['text']
#data Sentences 
data_words=['Study','Survey','Statistics','Data','Dataset','Database','census','Census']
sentences=nltk.sent_tokenize(full_text)
data_sentences=[w for w in sentences for i in range(0,len(data_words)) if data_words[i] in w]

#free from punctuation and pharantheses
tokenizer=RegexpTokenizer(r'\w+')
free_from_space=tokenizer.tokenize(' '.join(data_sentences))
#tokenized text
tokenized_text=nltk.word_tokenize(' '.join(free_from_space))
# normalization
norm_text=' '.join(tokenized_text).lower()

#Lemmatization
lemmatizer=nltk.WordNetLemmatizer()
lemmatized_text=[lemmatizer.lemmatize(w) for w in norm_text.split()]
#stopword_free text
stopwrds=stopwords.words('English')
stopword_free=[w for w in nltk.word_tokenize(' '.join(lemmatized_text)) if w not in stopwrds]
dictionary=corpora.Dictionary([stopword_free])
corpus=[dictionary.doc2bow(item) for item in [stopword_free]]
ldamodel=gensim.models.ldamodel.LdaModel(corpus,num_topics=1, id2word=dictionary,passes=25,per_word_topics=True)
ldamodel.print_topics(num_topics=1,num_words=5)[0][1]

#2Approach to detect datasources: Countvectorizer with n-grams and 
import nltk,re
import gensim
import json as js
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from gensim.models.ldamodel import LdaModel
#load text in json format
f_hand=open('2100032a-7c33-4bff-97ef-690822c43466.json','r')
text=js.load(f_hand)
#convert json to string
full_text=''
for i in range(0,len(text)):
    full_text=full_text+' '+text[i]['text']

# Eliminate stopwords 
free_from_stop=[w for w in nltk.word_tokenize(full_text) if w not in stopwords.words('English')]
# extract data sentences form text
data_words=['Study','Survey','Statistics','Dataset','Database','data','census','Census']
sentences=nltk.sent_tokenize(' '.join(free_from_stop))
data_sentences=[w for w in sentences for i in range(0,len(data_words)) if data_words[i] in w]
#print(data_sentences)
#countvectorizer and fit and transform text 
vectorizer=CountVectorizer(analyzer='word',ngram_range=(3,5))
fitted_model=vectorizer.fit_transform(data_sentences)
# create corpus to use in Lda model
corpus = gensim.matutils.Sparse2Corpus(fitted_model, documents_columns=False)
id_map = dict((v, k) for k, v in vectorizer.vocabulary_.items())

# Lda model
ldamodel=LdaModel(corpus,num_topics=1,id2word=id_map,passes=50,per_word_topics=True)
ldamodel.print_topics(num_topics=1,num_words=5)
