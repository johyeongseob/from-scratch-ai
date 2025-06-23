import glob
import unicodedata
import string
import os


# Tutorial1 step1
def findFiles(path):
    return glob.glob(path)

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


# Tutorial1 step2
def unicodeToAscii(string):
    # 1. 문자열을 유니코드 정규화
    normalized = unicodedata.normalize('NFD', string)

    # 만약 문자가 허용된 범위(all_letters)에 있고, 발음 구별 기호(Mn)가 아니면 추가
    ascii_characters = []
    for character in normalized:
        if (character in all_letters) and (unicodedata.category(character) != 'Mn'):
            ascii_characters.append(character)

    # 4. 리스트를 문자열로 변환 후 반환
    return ''.join(ascii_characters)


# Tutorial1 step3
category_lines = {}
all_categories = []
def readLines(filename):
    # 파일 내용을 읽어서 줄 단위로 나눔
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    # 각 줄을 유니코드에서 ASCII로 변환한 결과를 반환
    return [unicodeToAscii(line) for line in lines]

# 'data/names/' 디렉토리에서 모든 '.txt' 파일을 찾음
for filename in findFiles('../data/names/*.txt'):
    # 파일명에서 카테고리(언어) 이름 추출 (예: 'English.txt' → 'English')
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

if __name__ == "__main__":
    # 'data/names/' 디렉토리에서 확장자가 '.txt'인 모든 파일을 찾고 출력
    print(findFiles('../data/names/*.txt'))

    print(f"\nall_letters: {all_letters}")
    print(f"n_letters: {n_letters}")
    print(f"unicodeToAscii('Ślusàrski'): {unicodeToAscii('Ślusàrski')}\n")

    # 각 카테고리의 이름 개수와 첫 번째 이름을 출력
    total = 0
    print(f"all_categories: {all_categories}\n")

    for category in all_categories:
        print(f"category: {category}, num of names: {len(category_lines[category])}, first name: {category_lines[category][0]}")
        total += len(category_lines[category])
    print(f"\nTotal num of name: {total}")
    print(f"\nFive Italian names: {category_lines['Italian'][:5]}")