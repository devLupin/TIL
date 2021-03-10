"""
    SubwordTextEncoder
        - 텐서플로우의 서브워드 토크나이저
        - Wordpiece Model 채택(BPE와 유사)
        - 훈련 데이터에 등장하지 않은 단어는 각각 분리됨.
        - 기존 훈련 데이터에 없는 단어는 음절 이하 단위로 분리하고, 또한 정상적으로 디코딩
"""




""" IMDB reviews tokenization """
import tensorflow_datasets as tfds
import urllib.request
import pandas as pd

urllib.request.urlretrieve("https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv", \
                            filename="IMDb_Reviews.csv")

train_df = pd.read_csv('IMDb_Reviews.csv')

# 'review'열이 토큰화를 수행해야 할 데이터
# build_from_corpus를 통해서 서브워드들로 이루어진 단어 집합을 생성하고 각 서브워드에 고유한 정수 부여
tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    train_df['review'], target_vocab_size=2**13)

print(train_df['review'][20])
print('Tokenized sample question: {}'.format(tokenizer.encode(train_df['review'][20])))     # 정수 인코딩 수행 결과

# train_df에 존재하는 문장 중 일부를 발췌
sample_string = "It's mind-blowing to me that this film was even made."

# 인코딩한 결과를 tokenized_string에 저장
tokenized_string = tokenizer.encode(sample_string)
print ('정수 인코딩 후의 문장 {}'.format(tokenized_string))

# 이를 다시 디코딩
original_string = tokenizer.decode(tokenized_string)
print ('기존 문장: {}'.format(original_string))

print('단어 집합의 크기(Vocab size) :', tokenizer.vocab_size)

for ts in tokenized_string:
    print ('{} ----> {}'.format(ts, tokenizer.decode([ts])))




""" Naver movie reviews tokenization """
import tensorflow_datasets as tfds
import urllib.request

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", \
                            filename="ratings_train.txt")

train_data = pd.read_table('ratings_train.txt')

# Null 값이 존재하는 행 제거
if train_data.isnull().values.any():
    train_data = train_data.dropna(how = 'any')

# 인자로 네이버 영화 리뷰 데이터를 넣어서, 서브워드들로 이루어진 단어 집합 생성, 각 서브워드에 고유한 정수 부여
tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(train_data['document'], target_vocab_size=2**13)


# 임의로 21번째 선택
sample_string = train_data['document'][21]

# 인코딩한 결과를 tokenized_string에 저장
tokenized_string = tokenizer.encode(sample_string)
print ('정수 인코딩 후의 문장 {}'.format(tokenized_string))

# 이를 다시 디코딩
original_string = tokenizer.decode(tokenized_string)
print ('기존 문장: {}'.format(original_string))