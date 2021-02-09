"""
    scikit-learn을 이용한 LDA
"""

from numpy.lib.function_base import vectorize
import pandas as pd
import urllib.request
# urllib.request.urlretrieve("https://raw.githubusercontent.com/franciscadias/data/master/abcnews-date-text.csv", filename="abcnews-date-text.csv")
data = pd.read_csv('abcnews-date-text.csv', error_bad_lines=False)

print(len(data))
print(data.head(5))

text = data[['headline_text']]


### 텍스트 전처리 ###
import nltk

# 단어 토큰화
text['headline_text'] = text.apply(lambda row: nltk.word_tokenize(row['headline_text']), axis=1)
print(text.head(5))


# stopword 제거
from nltk.corpus import stopwords

stop = stopwords.words('english')
text['headline_text'] = text['headline_text'].apply(lambda x: [word for word in x if word not in (stop)])

print(text.head(5))


# 표제어 추출
# 3인칭 단수 표현 -> 1인칭 단수 표현, 과거 현재형 동사 -> 현재형
from nltk.stem import WordNetLemmatizer

text['headline_text'] = text['headline_text'].apply(lambda x: [WordNetLemmatizer().lemmatize(word, pos='v') for word in x])
print(text.head(5))


# 길이가 3이하인 단어 제거
tokenized_doc = text['headline_text'].apply(lambda x: [word for word in x if len(word) > 3])
print(tokenized_doc[:5])



### TF-IDF 행렬 ###

# 역토큰화
detokenized_doc = []
for i in range(len(next)) :
    t = ' '.join(tokenized_doc[i])
    detokenized_doc.append(t)
    
text['headline_text'] = detokenized_doc


# TF-IDF 행렬
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)   # 상위 1,000개 단어

X= vectorizer.fit_transform(text['headline_next'])      # 각 단어의 빈도수 기록


# LDA
from sklearn.decomposition import LatentDirichletAllocation
lda_model=LatentDirichletAllocation(n_components=10,learning_method='online',random_state=777,max_iter=1)

lda_top = lda_model.fit_transform(x)

terms = vectorizer.get_feature_names()  # 단어 집합 1000개 저장

def get_topics(components, feature_names, n=5) :
    for idx, topic in enumerate(components) :
        print("Topic %d:" % (idx+1), [(feature_names[i], topic[i].round(2)) for i in topic.argsort()[:-n - 1:-1]])
    
get_topics(lda_model.components_,terms)