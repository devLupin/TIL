#%%
from gensim.utils import tokenize
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt
# %%
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")
# %%
train_data = pd.read_table('ratings.txt')
# %%
train_data[:5] # 상위 5개 출력
# %%
print(len(train_data))
print(train_data.isnull().values.any())
# %%
train_data = train_data.dropna(how = 'any') # Null 값이 존재하는 행 제거
print(train_data.isnull().values.any()) # Null 값이 존재하는지 확인
# %%
# 불용어 정의
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
# %%
okt = Okt()
tokenized_data = []
for sentence in train_data['document'] :
    temp_X = okt.morphs(sentence, stem=True)    # Tokenization
    temp_X = [word for word in temp_X if not word in stopwords]     # remove stop_word
    tokenized_data.append(temp_X)
# %%
# 리뷰 길이 분포 확인
print('리뷰의 최대 길이 :',max(len(l) for l in tokenized_data))
print('리뷰의 평균 길이 :',sum(map(len, tokenized_data))/len(tokenized_data))
plt.hist([len(s) for s in tokenized_data], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()
# %%
from gensim.models import Word2Vec
model = Word2Vec(sentences = tokenized_data, size = 100, window = 5, min_count = 5, workers = 4, sg = 0)
model.wv.vectors.shape  # 완성된 임베딩 매트릭스의 크기 확인

print(model.wv.most_similar("최민식"))