"""
    Senetencepiece
        - BPE를 포함한 서브워드 토크나이징 알고리즘을 내장한 최선의 선택지
        - 사전 토큰화 작업 없이 전처리를 하지 않은 데이터에 바로 단어 분리 토큰화 가능
        - https://github.com/google/sentencepiece
"""

# pip install sentencepiece




""" IMDB reviews tokenization """
import sentencepiece as spm
import pandas as pd
import urllib.request
import csv

"""urllib.request.urlretrieve("https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv", \
                            filename="IMDb_Reviews.csv")"""

train_df = pd.read_csv('IMDb_Reviews.csv')
print(train_df['review'])

# for sentencepiece input
with open('imdb_review.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(train_df['review']))

"""
SentencePieceTrainer.Train() 인자
    - input : 학습시킬 파일
    - model_prefix : 만들어질 모델 이름
    - vocab_size : 단어 집합의 크기
    - model_type : 사용할 모델 (unigram(default), bpe, char, word)
    - max_sentence_length: 문장의 최대 길이
    - pad_id, pad_piece: pad token id, 값
    - unk_id, unk_piece: unknown token id, 값
    - bos_id, bos_piece: begin of sentence token id, 값
    - eos_id, eos_piece: end of sequence token id, 값
    - user_defined_symbols: 사용자 정의 토큰
"""
spm.SentencePieceTrainer.Train('--input=imdb_review.txt --model_prefix=imdb --vocab_size=5000 --model_type=bpe --max_sentence_length=9999')
# 완료되면 .model .vocab 파일이 생성됨.

"""
    - sep : 자료 구분 기준
    - header : 기본적으로 첫번째 행을 header로 지정. 불러올 header가 없는 경우 None 옵션 추가해야함.
"""
vocab_list = pd.read_csv('imdb.vocab', sep='\t', header=None, quoting=csv.QUOTE_NONE)
vocab_list.sample(10)

len(vocab_list)     # 상위의 --vocab_size와 동일

# load model file
sp = spm.SentencePieceProcessor()
vocab_file = "imdb.model"
sp.load(vocab_file)

lines = [
    "I didn't at all think of it this way.",
    "I have waited a long time for someone to film"
]
for line in lines:
    print(line)
    print(sp.encode_as_pieces(line))    # 문장 입력 시 서브 워드 시퀸스로 변환
    print(sp.encode_as_ids(line))       # 문장 입력 시 정수 시퀸스로 변환
    print()
    
sp.GetPieceSize()   # 단어 집합 크기 확인

print(sp.IdToPiece(430))   # 정수로부터 맵핑되는 서브 워드로 변환
print(sp.PieceToId('▁character'))   # 서브 워드로부터 맵핑되는 정수로 변환

sp.DecodeIds([41, 141, 1364, 1120, 4, 666, 285, 92, 1078, 33, 91])  # 정수 시퀸스로부터 문장으로 변환
sp.DecodePieces(['▁I', '▁have', '▁wa', 'ited', '▁a', '▁long', '▁time', '▁for', '▁someone', '▁to', '▁film'])     # 서브워드 시퀸스로부터 문장으로 변환

# 문장으로부터 인자값에 따라 정수 또는 서브워드 시퀸스로 변환 가능
print(sp.encode('I have waited a long time for someone to film', out_type=str))
print(sp.encode('I have waited a long time for someone to film', out_type=int))




""" Naver Movie reviews tokenization """
import pandas as pd
import sentencepiece as spm
import urllib.request
import csv

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", \
                            filename="ratings.txt")
naver_df = pd.read_table('ratings.txt')
print(naver_df[:5])

print(len(naver_df))

if naver_df.isnull().values.any():
    naver_df = naver_df.dropna(how = 'any') # Null 값이 존재하는 행 제거

# txt 파일에 저장 후, 센텐스피스를 통해 단어 집합 생성
with open('naver_review.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(naver_df['document']))
    
spm.SentencePieceTrainer.Train('--input=naver_review.txt --model_prefix=naver --vocab_size=5000 --model_type=bpe --max_sentence_length=9999')

vocab_list = pd.read_csv('naver.vocab', sep='\t', header=None, quoting=csv.QUOTE_NONE)

sp = spm.SentencePieceProcessor()
vocab_file = "naver.model"
sp.load(vocab_file)

lines = [
    "뭐 이딴 것도 영화냐.",
    "진짜 최고의 영화입니다 ㅋㅋ",
]
for line in lines:
    print(line)
    print(sp.encode_as_pieces(line))
    print(sp.encode_as_ids(line))
    print()
    
sp.GetPieceSize()

sp.IdToPiece(4)
sp.PieceToId('영화')

sp.DecodeIds([54, 200, 821, 85])
sp.DecodePieces(['▁진짜', '▁최고의', '▁영화입니다', '▁ᄏᄏ'])

print(sp.encode('진짜 최고의 영화입니다 ㅋㅋ', out_type=str))
print(sp.encode('진짜 최고의 영화입니다 ㅋㅋ', out_type=int))