"""
    Lemmatization(표제어 추출), Stemming(어간 추출) : 정규화 기법 중 단어의 개수를 줄일 수 있는 기법
    
    1. Lemmatization
        - 단어들로부터 표제어를 찾아가는 과정
        - 단어들이 다른 형태를 지녔어도, 뿌리 단어를 찾아가 단어의 개수를 줄일 수 있는지 판단
        - am, are, is 의 표제어는 be
        
        - 표제어 추출은 단어의 형태학(형태소로부터 단어들을 만듦)적 파싱을 우선 진행하는 것
        - 형태소 종류
            1. stem(어간) : 단어의 의미를 담고 있는 핵심 부분
            2. affix(접사) : 단어에 추가적인 의미를 주는 부분
            ex) cats ==> cat(stem), s(affix)
        - 형태학적 파싱은 stem과 affix를 분리하는 작업
        - 문맥을 고려하여, 수행했을 때의 결과는 해당 단어의 품사 정보를 보존(POS 태그 보존)
        
    2. Stemming
        - 형태학적 분석의 단순화한 버전
        - 정해진 규칙만 보고 단어의 어미를 자르는 어림짐작의 작업
        - 어간 추출을 수행한 결과는 품사 정보가 보존되지 않음.(POS 태그 보존 X)
        - 섬세하지 못한 작업이므로 사전에 존재하지 않는 단어일 경우가 많음.
"""


## Lemmatization

from nltk.stem import WordNetLemmatizer     # WordNetLemmatize : 표제어을 위한 도구 지원

n = WordNetLemmatizer()
word = ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']

print([n.lemmatize(w) for w in word])



## Stemming

# Poter algorithm

from nltk.stem import PorterStemmer     # Poter Algorithm is one of the best Stem algorithm
from nltk.tokenize import word_tokenize

s = PorterStemmer()

text = "This was not the map we found in Billy Bones's chest, but an accurate copy, complete in all things--names and heights and soundings--with the single exception of the red crosses and the written notes."
words = word_tokenize(text)

print(words)

words = ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
print([s.stem(w) for w in words])

# Lancaster stemmer algorithm

from nltk.stem import LancasterStemmer
l = LancasterStemmer()

print([l.stem(w) for w in words])

"""
['polici', 'do', 'organ', 'have', 'go', 'love', 'live', 'fli', 'die', 'watch', 'ha', 'start']       # Poter algorithm
['policy', 'doing', 'org', 'hav', 'going', 'lov', 'liv', 'fly', 'die', 'watch', 'has', 'start']     # Lancaster stemmer algorithm

결과가 서로 상이하므로 사용하고자 하는 코퍼스에 스태머를 적용해보고 어떤 스태머가 해당 코퍼스에 적합한지 판단 후에 사용해야 함.

어간 추출 후 일반화가 지나치게 되거나 덜 되거나 하는 경우 규칙에 기반한 알고리즘은 종종 제대로된 일반화를 수행하지 못함.
"""