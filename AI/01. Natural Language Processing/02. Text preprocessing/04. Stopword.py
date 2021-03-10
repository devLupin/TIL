"""
    Stopword(불용어)
        - 유의미한 단어 토큰만을 선별하기 위해서는 큰 의미가 없는 단어 토큰을 제거하는 작업 필요
        - 실제 의미 분석을 하는데 거의 기여하는 바가 없는 단어
        - NLTK에서는 100여개의 불용어를 패키지 내에서 정의
"""

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# print stopword list

stopword_list = stopwords.words('english')[:10]     # 0~9 list
print(stopword_list)


# Example for remove stopword in English

example = "Family is not an important thing. It's everything."
stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(example)

result = []
for w in word_tokens:
    if w not in stop_words:
        result.append(w)
        
print(word_tokens)
print(result)

# Example for remove stopword in Korean

"""
    토큰화 후 조사, 접속사 제거
    대부분 사용자가 직접 불용어 사전을 정의하여 제거
    일반적인 경우의 불용어 사전은 txt, csv 파일로 정리
    
    보편적으로 선택할 수 있는 한국어 불용어 리스트
        - https://www.ranks.nl/stopwords/korean
"""

example = "고기를 아무렇게나 구우려고 하면 안 돼. 고기라고 다 같은 게 아니거든. 예컨대 삼겹살을 구울 때는 중요한 게 있지."
stop_words = "아무거나 아무렇게나 어찌하든지 같다 비슷하다 예컨대 이럴정도로 하면 아니거든"

stop_words = stop_words.split(' ')      # split on a blank basis
word_tokens = word_tokenize(example)

result = []
result=[word for word in word_tokens if not word in stop_words]     # == code line 25~27

print(word_tokens)
print(result)