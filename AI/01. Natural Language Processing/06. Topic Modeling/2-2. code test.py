"""
    gensim을 이용한 LDA
"""


from funcy.flow import ignore
from gensim.models import ldamodel
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data
# len(documents)  # 샘플 수 출력
# print(dataset.target_names)     # 카테고리 종류 출력

news_df = pd.DataFrame({'document':documents})

news_df['clean_doc'] = news_df['document'].str.replace("[^a-zA-Z]", " ")    # 특수 문자 제거
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))   # 길이가 3이하인 단어는 제거 (길이가 짧은 단어 제거)
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower())      # 전체 단어에 대한 소문자 변환

from nltk.corpus import stopwords

stop_words = stopwords.words('english')
tokenized_doc = news_df['clean_doc'].apply(lambda x: x.split()) # 토큰화
tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])   # # 불용어 제거

from gensim import corpora

dictionary = corpora.Dictionary(tokenized_doc)      # 정수 인코딩 번호에 따른 단어들
corpus = [dictionary.doc2bow(text) for text in tokenized_doc]   # corpus는 [(정수 인코딩, 빈도 수)]의 형태

import gensim

NUM_TOPICS = 20     # 토픽의 개수 k=20
lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=1)
topics = lda_model.print_topics(num_words=4)

# for topic in topics :
    # print(topic)
    # (0, '0.015*"drive" + 0.014*"thanks" + 0.012*"card" + 0.012*"system"')
"""
    단어 앞에 붙은 수치는 해당 토픽에 대한 기여도
    토픽 번호는 단순히 0~19
    passes는 알고리즘의 동작 횟수, 토픽의 값이 적절히 수렴할 수 있도록 적당한 횟수 지정
    num_words=4로 4개의 단어만 출력
"""

### 토픽별 단어 분포 시각화 코드, 노트북만 사용 가능함
"""
import pyLDAvis.gensim
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
pyLDAvis.display(vis)
"""

### 문서별 토픽 분포
for i, topic_list in enumerate(lda_model[corpus]) :   # 전체 데이터 정수 인코딩 된 결과 삽입
    if i==5 :
        break
    # print(i,'번째 문서의 topic 비율은',topic_list)

def make_topictalbe_per_doc(ldamodel, corpus) :
    topic_table = pd.DataFrame()
    
    for i, topic_list in enumerate(ldamodel[corpus]) :  # (문서 번호, 토픽 비중)
        doc = topic_list[0] if ldamodel.per_word_topics else topic_list
        doc = sorted(doc, key=lambda x: (x[1]), reverse=True)   # 각 문서에 대해서 비중이 높은 토픽 순으로 정렬
        
        # 모든 문서에 대해 반복
        for j, (topic_num, prop_topic) in enumerate(doc) :  # 몇 번 토픽인지, 비중을 나눠서 저장
            if j == 0:      # 가장 비중이 높은 토픽이라면
                # 가장 비중이 높은 토픽, 가장 비중이 높은 토픽의 비중, 전체 토픽의 비중을 저장
                topic_table = topic_table.append(pd.Series([int(topic_num), round(prop_topic, 4), topic_list]), ignore_index=True)
            else :
                break
            
    return (topic_table)

topic_table = make_topictalbe_per_doc(ldamodel, corpus)
topic_table = topic_table.reset_index()     # 인덱스 열을 하나 더 만든다.
topic_table.columns = ['문서 번호', '가장 비중이 높은 토픽', '가장 높은 토픽의 비중', '각 토픽의 비중']
print(topic_table[:10])