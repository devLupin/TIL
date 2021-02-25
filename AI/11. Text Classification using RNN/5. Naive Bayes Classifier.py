from sklearn.datasets import fetch_20newsgroups
newsdata = fetch_20newsgroups(subset='train')
print(newsdata.keys())
#print(newsdata.target_names)
#print(newsdata.data[0])

""" preprocessing of newsdata.data """
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB # 다항분포 나이브 베이즈 모델
from sklearn.metrics import accuracy_score #정확도 계산

dtmvector = CountVectorizer()
X_train_dtm = dtmvector.fit_transform(newsdata.data)
#print(X_train_dtm.shape)    # (num of train sample, num of all train word)

"""
    TF-IDF 행렬을 입력으로 텍스트 분류를 수행하면, 성능의 개선을 얻을 수도 있음.
    항상 DTM 보다 성능이 뛰어나진 않음.
"""
tfidf_transformer = TfidfTransformer()
tfidfv = tfidf_transformer.fit_transform(X_train_dtm)
#print(tfidfv.shape)

mod = MultinomialNB     # 나이브 베이즈 모델
mod.fit(tfidfv, newsdata.target)    # (X_train, y_train)

MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
""" alpha=1.0은 라플라스 스무딩이 적용되었음을 의미 """

newsdata_test = fetch_20newsgroups(subset='test', shuffle=True) #테스트 데이터 갖고오기
X_test_dtm = dtmvector.transform(newsdata_test.data) #테스트 데이터를 DTM으로 변환
tfidfv_test = tfidf_transformer.transform(X_test_dtm) #DTM을 TF-IDF 행렬로 변환

predicted = mod.predict(tfidfv_test) #테스트 데이터에 대한 예측
print("정확도:", accuracy_score(newsdata_test.target, predicted)) #예측값과 실제값 비교