"""
    Integer Encoding(정수 인코딩)
        - 텍스트를 숫자로 바꾸는 단계 중 첫 단계
        - 각 단어를 고유한 정수에 mapping시키는 전처리 작업
        - 일반적으로 단어에 대한 빈도수를 기준으로 정렬한 뒤에 부여
        
        - 텍스트 데이터 중 빈도수가 가장 높은 n개의 단어만 사용하는 경우가 많음.
        - n개의 단어 집합에 포함되지 않는 단어를 Out-Of-Vocabulary(OOV, 단어 집합에 없는 단어)
        - 정수 인코딩 집합에 'OOV'란 단어를 추가하여, OOV는 'OOV'의 인덱스로 인코딩함.
        
        - Counter, FreqDist, enumerate, Keras toknizer 사용이 권장됨.
"""



### Use dictionary ###

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

text = "A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret He Kept is huge secret. Huge secret. His barber kept his word. a barber kept his word. His barber kept his secret. But keeping and keeping such a huge secret to himself was driving the barber crazy. the barber went up a huge mountain."

text = sent_tokenize(text) # sentence tokenize

vocab = {}
sentences = []
stop_words = set(stopwords.words('english'))

for i in text:
    sentence = word_tokenize(i) # word tokenize
    result = []
    
    for word in sentence :
        word = word.lower() # All voca changed lowercase. Because of prevent duplicate count
        
        if word not in stop_words:  # Remove stop word
            if len(word) > 2 :       # Additional, remove word if word length is 2 or less
                result.append(word)
                
                if word not in vocab :
                    vocab[word] = 0
                    
                vocab[word] += 1    # Counting frequency
                
    sentences.append(result)

"""
    lambda x:x[1]
        - x의 원소를 x의 1번째 원소와 비교하여 정렬한다
"""
vocab_sorted = sorted(vocab.items(), key = lambda x:x[1], reverse = True)   # Sort by frequency cnt

"""     # 상위 n개 단어만 사용하고 싶은 경우 주석 제거
    words_frequency = [w for w,c in word_to_index.items() if c >= vocab_size + 1]   # 인덱스가 5 초과인 단어 제거
    
"""

word_to_index = {}
i = 0
for (word, frequency) in vocab_sorted :
    if frequency > 1 :
        i = i + 1
        word_to_index[word] = i
        
word_to_index['OOV'] = len(word_to_index) + 1   # OOV index encoding


# Integer encoding

encoded = []

for s in sentences :
    temp = []
    
    for w in s :
        try:
            temp.append(word_to_index[w])
        except KeyError:
            temp.append(word_to_index['OOV'])
    
    encoded.append(temp)
    
print(encoded)



### Use counter ###

from collections import Counter

"""
    sentences는 단어 토큰화 된 결과가 저장되어 있음.
    [['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]
    
    이를 단어 집합으로 만들기 위해서 문장의 경계인 [, ]를 제거하여 하나의 리스트로
"""
words = sum(sentences, [])  # == np.hstack(sentences)

vocab = Counter(words)

vocab_size = 5
vocab = vocab.most_common(vocab_size)   # save word that higher frequency


# A word with a higher frequency is given a lower integer index.

i = 0
for (word, frequency) in vocab :
    i = i+1
    word_to_index[word] = i
    


### Use FreqDist() of NLTK ###

from nltk import FreqDist
import numpy as np

vocab = FreqDist(np.hstack(sentences))  # sentences의 데이터를 배열 형태로 수평으로 쌓는다.
vocab = vocab.most_common(vocab_size)

"""
    enumerate
        - 순서가 있는 자료형을 입력으로 받아 인덱스를 순차적으로 함께 리턴
"""
word_to_index = {word[0] : index + 1 for index, word in enumerate(vocab)}   # == 109 ~ 111 lines



### Use Keras toknizer ###

from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)   # If input corpus, Generate word set based frequency.

word_counts = tokenizer.word_counts     # number of each word frequency

text_to_sequence = tokenizer.texts_to_sequences(sentences)  # Integer encoding


# Use words with higher frequency

# Keras Tokenizer는 OOV에 대해서 아예 단어를 제거하는 특성을 지님.
# 이를 보존하고 싶다면 아래 주석 해제
tokenizer = Tokenizer(num_words = vocab_size + 1)
"""
    tokenizer = Tokenizer(num_words = vocab_size + 2, oov_token = 'OOV')    # OOV를 고려하여 +2

    # OOV의 인덱스는 1이다.
"""

tokenizer.fit_on_texts(sentences)

tokenizer.texts_to_sequences(sentences)