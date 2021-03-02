"""
    개체명 인식 : 코퍼스로부터 각 개체(entity)의 유형을 인식
    
    NLTK에서는 개체명 인식기(NER chunker)를 지원
"""
from nltk import word_tokenize, pos_tag, ne_chunk

sentence = "James is working at Disney in London"
sentence = pos_tag(word_tokenize(sentence)) # 토큰화와 품사 태깅을 동시 수행

# 개체명 태깅(ne_chunk)을 하기 앞서 품사 태깅(pos_tag)이 수행되어야 함.
sentence = ne_chunk(sentence)
print(sentence)