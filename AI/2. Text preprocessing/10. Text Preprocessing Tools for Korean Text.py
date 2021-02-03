"""
    Text Preprocessing Tools for Korean Text(한국어 전처리 패키지)
        - 형태소와 문장 토크나이징 도구인 KoNLPy, KSS(Korean Sentence Splitter)와 함께 유용하게 사용할 수 있는 패키지
"""



### PyKoSpacing ###
# 띄어쓰기가 되어있지 않은 문장을 띄어쓰기 한 문장으로 변환해주는 패키지
# pip install git+https://github.com/haven-jeon/PyKoSpacing.git

sent = '김철수는 극중 두 인격의 사나이 이광수 역을 맡았다. 철수는 한국 유일의 태권도 전승자를 가리는 결전의 날을 앞두고 10년간 함께 훈련한 사형인 유연재(김광수 분)를 찾으러 속세로 내려온 인물이다.'
new_sent = sent.replace(" ", '') # 띄어쓰기가 없는 문장 임의로 만들기

from pykospacing import spacing
kospacing_sent = spacing(new_sent)      # sent == kospacing_sent



### Py-Hanspell ###
# 네이버 한글 맞춤법 검사기를 바탕으로 만들어진 패키지, 띄어쓰기 또한 보정
# pip install git+https://github.com/ssut/py-hanspell.git

from hanspell import spell_checker

sent = "맞춤법 틀리면 외 않되? 쓰고싶은대로쓰면돼지 "
spelled_sent = spell_checker.check(sent)
"""
    print(spelled_sent)
    
    ==> Checked(result=True, original='맞춤법 틀리면 외 않되? 쓰고싶은대로쓰면돼지 ', checked='맞춤법 틀리면 왜 안돼? 쓰고 싶은 대로 쓰면 되지', errors=2, words=OrderedDict([('맞춤법', 0), ('틀리면', 0), ('왜', 1), ('안돼?', 1), ('쓰고', 1), ('싶은', 1), ('대로', 1), ('쓰면', 1), ('되지', 1)]), time=0.0850071907043457)
"""
hanspell_sent = spelled_sent.checked



### SOYNLP ###
"""
    - 품사 태깅, 단어 토큰화 등을 지원하는 단어 토크나이저
    - 비지도 학습으로 단어 토큰화
    - 데이터에 자주 등장하는 단어들을 단어로 분석
    - 내부적으로 단어 점수표로 동작. 이 점수는 응집 확률(cohension probability)과 브랜칭 엔트로피(branching entropy)를 활용
    
    - 응집 확률(cohension probability)
        a. 응집 확률은 내부 문자열이 얼마나 응집하여 자주 등장하는지 판단하는 척도
        b. 응집 확률은 문자열을 문자 단위로 분리하여 내부 문자열을 만드는 과정에서 왼쪽부터 순서대로 문자를 추가하면서 각 문자열이 주어졌을 때 그 다음 문자가 나올 확률을 계산하여 누적곱 한 값
        c. 이 값이 높을수록 전체 코퍼스에서 하나의 단어로 등장할 가능성이 높음.
    - 브랜칭 엔트로피(branching entropy)
        a. 확률 분포의 엔트로피 값 사용. 이는 주어진 문자열에서 다음 문자가 등장할 수 있는지 판단하는 척도임.
        b. 브랜칭 엔트로피 값은 하나의 완성된 단어에 가까워질수록 문맥으로 인해 점점 정확히 예측할 수 있게 되면서 점점 줄어드는 양상임.
    
    - 텍스트 데이터에서 특정 문자 시퀀스가 함께 자주 등장하는 빈도가 높고, 앞 뒤로 조사 또는 완전히 다른 단어가 등장하는 것을 고려해서 해당 문자 시퀀스를 형태소라고 판단하는 단어 토크나이저
    - 신조어에도 어느 정도 대비 가능
"""

import urllib.request
from soynlp import DoublespaceLineCorpus
from soynlp.word import WordExtractor

urllib.request.urlretrieve("https://raw.githubusercontent.com/lovit/soynlp/master/tutorials/2016-10-20.txt", filename="2016-10-20.txt")     # 한국어 문서 다운로드

corpus = DoublespaceLineCorpus("2016-10-20.txt")
len(corpus)     # 30091개의 문서 존재
type(corpus)    # <class 'soynlp.utils.utils.DoublespaceLineCorpus'>

"""
    soynlp는 학습 기반의 단어 토크나이저이므로 기존의 KoNLPy에서 제공하는 형태소 분석기들과는 달리 학습 과정을 거쳐야 함. 
    이는 전체 코퍼스로부터 응집 확률과 브랜칭 엔트로피 단어 점수표를 만드는 과정
"""
word_extractor = WordExtractor()    # 전체 코퍼스에 대해 단어 점수표 계산
word_extractor.train(corpus)
word_score_table = word_extractor.extract()
"""
>>
    training was done. used memory 1.995 Gb
    all cohesion probabilities was computed. # words = 223348
    all branching entropies was computed # words = 361598
    all accessor variety was computed # words = 361598
"""

# 응집 확률(cohesion probability)
word_score_table["반포한강공원"].cohesion_forward   # "반포한강공원에" 보다 낮음
word_score_table["반포한강공원에"].cohesion_forward

# 브랜칭 엔트로피(branching entropy)
word_score_table["디스"].right_branching_entropy    # >> 1.6371694761537934
word_score_table["디스플"].right_branching_entropy  # >> -0.0   // 레가 오는 것이 너무 명백함.
word_score_table["디스플레이"].right_branching_entropy # >> 3.1400392861792916
"""
    디스플레이에서 갑자기 값 증가
    이는 문자 시퀸스인 디스플레이 다음에 조사나 다른 단어 같은 다양한 경우가 있을 수 있기 때문임.
    하나의 단어가 끝나면 그 경계 부분부터 다시 브랜칭 엔트로피 값이 증가하게 됨을 의미
"""


# L tokenizer
from soynlp.tokenizer import LTokenizer

scores = {word:score.cohesion_forward for word, score in word_score_table.items()}      # 단어와 점수로 딕셔너리에 저장
l_tokenizer = LTokenizer(scores=scores)     # L토큰 + R토큰으로 나누고, 분리 기준 점수가 가장 높은 L 토큰을 찾아내는 원리
l_tokenizer.tokenize("국제사회와 우리의 노력들로 범죄를 척결하자", flatten=False)
# >> ['국제사회', '와', '우리', '의', '노력', '들로', '범죄', '를', '척결', '하자']

# 반복되는 문자 정제
# ㅋㅋ, ㅎㅎ 같은 불필요하고 연속적인 것들을 정규화
# num_repeats 는 중복되는 글자를 최대 n개까지 수용하겠다 이런 뜻임.
from soynlp.normalizer import *
print(emoticon_normalize('앜ㅋㅋㅋㅋ이영화존잼쓰ㅠㅠㅠㅠㅠ', num_repeats=2))
print(emoticon_normalize('앜ㅋㅋㅋㅋㅋㅋㅋㅋㅋ이영화존잼쓰ㅠㅠㅠㅠ', num_repeats=2))
print(emoticon_normalize('앜ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ이영화존잼쓰ㅠㅠㅠㅠㅠㅠ', num_repeats=2))
print(emoticon_normalize('앜ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ이영화존잼쓰ㅠㅠㅠㅠㅠㅠㅠㅠ', num_repeats=2))


# Customized KoNLPy
# pip install customized_konlpy
from ckonlpy.tag import Twitter
twitter = Twitter()
twitter.morphs('은경이는 사무실로 갔습니다.')   # 이 때 '은'은 형용사 뒤에 쓰이는 것으로 인식되어 토큰화 되어 버림.
twitter.add_dictionary('은경이', 'Noun')    # 형태소 분석기 Twitter에 add_dictionary('단어', '품사')와 같은 형식으로 사전 추가를 해줄 수 있음.
twitter.morphs('은경이는 사무실로 갔습니다.')
"""
    add_dictionaly 전
        >> ['은', '경이', '는', '사무실', '로', '갔습니다', '.']
    add_dictionaly 후
        >> ['은경이', '는', '사무실', '로', '갔습니다', '.']
"""