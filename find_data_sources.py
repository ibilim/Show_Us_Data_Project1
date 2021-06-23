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
import re
import nltk
import json as js
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from gensim.models.ldamodel import LdaModel

def get_data_sources(text)
    #load text in json format
    f_hand=open(text,'r')
    text=js.load(f_hand)

    #convert json to string
    full_text=''
    for i in range(0,len(text)):
        full_text=full_text+' '+text[i]['text']
    #free from punctuation pharantheses and dates    
    # FIND CAPITAL LETTERS
    capital_letters=re.findall(r'([A-Z]{3,})',full_text)
    capitalletters=list(capital_letters)
    dict_capitalletters={}
    for i in capitalletters:
        dict_capitalletters[i]=dict_capitalletters.get(i,0)+1
    #dict_capitalletters
    #REPLACE CAPICAL LETTERS with long versions
    #here we had to remove stop words in order to find out long versions of datasources in text
    ffs=[w for w in nltk.word_tokenize(full_text) if w.lower() not in stopwords.words('English')]
    ffs_text=' '.join(ffs)
    for i in ffs:
        for cap_let in set(capitalletters):
            if i==cap_let:
                word_to_find=[cl+'[A-Za-z]+' for cl in cap_let]
                wrdtofind=' '.join(word_to_find)
                #print(wrdtofind)
                #print(re.search(wrdtofind,ffs_text))
                try:
                    ffs_text=re.sub(i,re.search(wrdtofind ,ffs_text)[0],ffs_text)
                    #ffs_text=re.sub(i,re.search(wrdtofind.lower(),ffs_text)[0],ffs_text)
                except:
                    continue
            else:
                continue

    #free from punctuation and pharantheses
    free_from_symb_dates=[]
    for i in nltk.sent_tokenize(ffs_text):
        free_from_symb_dates.append(re.sub(r'\b\w\w\b','',re.sub(r'\d+','',re.sub(r'\W+',' ', i)))+'.') # clear each sentence in ffs_text text 
                                                                                                        #from dates symbols and two letter words 
    #data Sentences 
    data_words= set(capitalletters)+['Study','Survey','Statistics','Data','Dataset','Database','Census','Assessment'] # words that include in the list 

    data_sentences=[w for w in free_from_symb_dates for i in range(0,len(data_words)) if data_words[i].lower() in w.lower()]
    #countvectorizer and fit and transform text 
    vectorizer=CountVectorizer(analyzer='word',max_df=0.7,ngram_range=(3,5))# ngram_range=(3,5) ,eliminates the words that exceeds max_df frequency
    fitted_model=vectorizer.fit_transform(set(data_sentences)) # fit and transform data sentences to vectorizer
    # create corpus to use in Lda model
    corpus = gensim.matutils.Sparse2Corpus(fitted_model, documents_columns=False) # creates main corpus to analyze
    id_map = dict((v, k) for k, v in vectorizer.vocabulary_.items()) # creates dictionary to be used in anlysis

    # Lda model
    ldamodel=LdaModel(corpus,num_topics=1,id2word=id_map,per_word_topics=True,passes=50) # Lda model to predict data sources
    ldamodel.print_topics(num_topics=1,num_words=5) # print topic that has the highest probability
    return ldamodel.show_topic(topicid=0,topn=4)[0][0]+'|'+ldamodel.show_topic(topicid=0,topn=4)[1][0]+'|'+ldamodel.show_topic(topicid=0,topn=4)[2][0]

header='Id,PredictionString'
file_open=open('data_in_article.txt','w')
file_open.write(header+"\n")
file_open.close()
text_ids=['2100032a-7c33-4bff-97ef-690822c43466',
           '2f392438-e215-4169-bebf-21ac4ff253e1',
           '3f316b38-1a24-45a9-8d8c-4e05a42257c6',
           '8e6996b4-ca08-4c0b-bed2-aaf07a4c6a60']
    
with open('data_in_article.txt', 'a') as f:
    for ids in text_ids:
        f.write(ids+ ", " + get_data_sources(ids+'.json')+"\n")
f.close()
