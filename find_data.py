import json as js
import nltk
import pandas as pd
fh=open('2f392438-e215-4169-bebf-21ac4ff253e1.json','r')
fh_1=js.load(fh)

text1=fh_1
full_text=''
for i in range(0,len(text1)):
    full_text=full_text+text1[i]['text']
full_text

data_words=['survey','study']
sentences=nltk.sent_tokenize(full_text)

data_sentences=[w for w in sentences if data_words[1] in w.lower() ]
data_sentences

#Noun-Phrase Approach to detect datasources
import re,pprint
sentence1=nltk.pos_tag(nltk.word_tokenize(data_sentences[0]),lang='eng')
grammar = "NP: {<DT>?<NNP>}"
cp=nltk.RegexpParser(grammar)
result=cp.parse(sentence1)
print(result)
#result.draw()
