import unicodedata
import re
import random


SOS_token = 0
EOS_token = 1

# 어휘 관리 클래스: 단어를 인덱스와 매핑하고, 단어 빈도를 추적
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}  # 단어 -> 인덱스 매핑
        self.word2count = {}  # 단어 -> 등장 횟수
        self.index2word = {0: "SOS", 1: "EOS"}  # 인덱스 -> 단어 매핑
        self.n_words = 2  # vocab size

    # 문장을 단어로 분리하여 각 단어를 사전에 추가
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)


    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words  # 단어에 새 인덱스 부여
            self.index2word[self.n_words] = word  # 인덱스 -> 단어 매핑
            self.word2count[word] = 1  # 단어 등장 횟수 초기화
            self.n_words += 1  # 단어 수 증가
        else:
            self.word2count[word] += 1  # 단어가 이미 있다면 등장 횟수 증가


# 유니 코드 문자열을 일반 ASCII로 변환
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# 소문자, 다듬기, 그리고 문자가 아닌 문자 제거
# re: Regular Expression, sub: Substitute, r"": raw string, \1: matching word
# re.sub(pattern, repl, string): pattern: 정규식 패턴, repl: 대체할 문자열, string: 변환할 대상 문자열.
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()

def readLangs(lang1, lang2, reverse=False):

    # 파일을 읽고 줄로 분리
    file = open(f'../data/{lang1}-{lang2}.txt', encoding='utf-8')
    lines = file.read().strip().split('\n')

    # 모든 줄을 쌍으로 분리하고 정규화
    pairs = []
    for line in lines:  # 파일의 각 줄에 대해 반복
        pair = line.split('\t')  # 탭(\t)을 기준으로 줄을 나눠 문장 쌍을 생성
        normalized_pair = []
        for sentence in pair:  # 문장 쌍의 각 문장에 대해 반복
            normalized_pair.append(normalizeString(sentence))  # 문장을 정규화 후 추가
        pairs.append(normalized_pair)  # 정규화된 문장 쌍을 리스트에 추가

    # 쌍을 뒤집고, Lang 인스턴스 생성
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


# 문장 내, 단어 수가 10개를 넘지 않아야 함
MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filterPair(pair):
    return len(pair[0].split(' ')) < MAX_LENGTH and \
        len(pair[1].split(' ')) < MAX_LENGTH and \
        pair[1].startswith(eng_prefixes)


def filterPairs(pairs):
    filtered_pairs = []
    for pair in pairs:
        if filterPair(pair):
            filtered_pairs.append(pair)
    return filtered_pairs


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    pairs = filterPairs(pairs)
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    return input_lang, output_lang, pairs


if __name__ == '__main__':

    # Example 1
    lang = Lang("English")
    input_str = "Hello, world!"
    lang.addSentence(input_str)
    print(f"\nExample 1, Lang name: {lang.name}, addSentence: {input_str}, word2index: {lang.word2index}, "
          f"word2count:{lang.word2count}, index2word: {lang.index2word}, n_words: {lang.n_words}")

    # Example 2
    print(f"Example 2, normalizeString({input_str}): {normalizeString(input_str)}")

    # Example 3
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    print(f"Example 3, Random pair [france -> english]: {random.choice(pairs)}")
