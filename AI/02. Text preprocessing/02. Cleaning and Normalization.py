"""
    정제(Cleaning), 정규화(Normalization)

        Tokenization : corpus에서 용도에 맞게 토큰을 분류하는 작업
        Tokenization 전, 후에는 텍스트 데이터를 용도에 맞게 정제 및 정규화 한다.

        Cleaning : 갖고 있는 corpus로부터 noise 데이터를 제거
        Normalization : 표현 방법이 다른 단어들을 통합시켜서 같은 단어로 만들어 줌.

    영어권 언어는 일반적인 경우, 소문자로 대, 소문자 통합 작업을 진행

    Removing Unnecessary Words(불필요한 단어 제거)
        noise data : 자연어가 아니면서 아무 의미도 갖지 않는 특수문자 등, 분석하고자 하는 목적에 맞지 않는 불필요 단어
        불용어 제거, 빈도가 적은 단어, 길이가 짧은 단어들을 제거하는 방법이 있음.
"""

import re # regular expression module

text = "I was wondering if anyone out there could enlighten me on this car."

shortword = re.compile(r'\W*\b\w{1,2}\b')    # 길이가 1~2인 단어들을 정규 표현식을 이용하여 삭제
print(shortword.sub('', text))