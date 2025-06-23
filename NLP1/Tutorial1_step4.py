"""

이름을 Tensor로 변경

    역자 주: One-Hot 벡터는 언어 및 범주형 데이터를 다룰 때 주로 사용하며,
    단어, 글자 등을 벡터로 표현할 때 단어, 글자 사이의 상관 관계를 미리 알 수 없을 경우,
    One-Hot으로 표현하여 서로 직교한다고 가정하고 학습을 시작합니다.
    이와 동일하게, 상관 관계를 알 수 없는 다른 데이터의 경우에도 One-Hot 벡터를 활용할 수 있습니다.

"""

import torch
from Tutorial1_step2 import all_letters, n_letters


# 문자열에서 특정 부분 문자열(sub)의 첫 번째 등장 위치(인덱스)를 반환
def letterToIndex(letter):
    return all_letters.find(letter)


# 한 개의 문자를 <1 x n_letters> Tensor로 변환
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


# 한 줄(이름)을  <line_length x 1 x n_letters>, 또는 One-Hot 문자 벡터의 Array로 변경
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


if __name__ == "__main__":
    print(f"\nletterToIndex('J'): {letterToIndex('J')}\n")
    print(f"letterToTensor('J'): {letterToTensor('J')}\n")
    print(f"lineToTensor('Jones'): {lineToTensor('Jones')}")