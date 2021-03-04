#%%
from nltk import tokenize
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from tensorflow.keras.preprocessing.text import Tokenizer

#%%
dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data
print('총 샘플 수 :',len(documents))

#%%
news_df = pd.DataFrame({'document' : documents})
news_df['clean_doc'] = news_df['document'].str.replace("[^a-zA-Z]", " ")    # 특수문자 제거
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))   # 길이가 3이하인 단어 제거
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower())  # 소문자로 변환
# %%
news_df.isnull().values.any()
# %%
news_df.replace("", float("NaN"), inplace=True)     # 모든 빈 값을 Null 값으로 변환
news_df.isnull().values.any()
# %%
news_df.dropna(inplace=True)    # remove null
print(len(news_df))
# %%
""" remove stop_word """
stop_words = stopwords.words('english')
tokenized_doc = news_df['clean_doc'].apply(lambda x: x.split())
tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
tokenized_doc = tokenized_doc.to_list()
# %%
drop_train = [index for index, sentence in enumerate(tokenized_doc) if len(sentence) <= 1]  # 단어가 1개 이하인 샘플의 인덱스를 찾아서 저장
tokenized_doc = np.delete(tokenized_doc, drop_train, axis=0)    # 해당 샘플 삭제
# %%
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tokenized_doc)   # 단어 집합 생성

# 정수 인코딩 
word2idx = tokenizer.word_index
idx2word = {v:k for k, v in word2idx.items()}
encoded = tokenizer.texts_to_sequences(tokenized_doc)
# %%
print(encoded[:2])
# %%
vocab_size = len(word2idx) + 1 
print(vocab_size)
# %%
from tensorflow.keras.preprocessing.sequence import skipgrams   # for negative sampling
skip_grams = [skipgrams(sample, vocabulary_size=vocab_size, window_size=10) for sample in encoded[:10]]
# %%
pairs, labels = skip_grams[0][0], skip_grams[0][1]
for i in range(5):
    print("({:s} ({:d}), {:s} ({:d})) -> {:d}".format(
          idx2word[pairs[i][0]], pairs[i][0], 
          idx2word[pairs[i][1]], pairs[i][1], 
          labels[i]))
# %%
print(len(pairs))
print(len(labels))
# %%
skip_grams = [skipgrams(sample, vocabulary_size=vocab_size, window_size=10) for sample in encoded]
# %%
print('전체 샘플 수 :',len(skip_grams))
# %%
print(len(pairs))
print(len(labels))