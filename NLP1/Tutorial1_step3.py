from Tutorial1_step1 import findFiles
from Tutorial1_step2 import unicodeToAscii
import os


category_lines = {}
all_categories = []

# 파일을 읽고 줄 단위로 분리하는 함수
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
    # 각 카테고리의 이름 개수와 첫 번째 이름을 출력
    total = 0
    for category in all_categories:
        print(f"category: {category}, num of names: {len(category_lines[category])}, first name: {category_lines[category][0]}")
        total += len(category_lines[category])
    print(f"\nTotal num of name: {total}")

    print(f"\nFive Italian names: {category_lines['Italian'][:5]}")
