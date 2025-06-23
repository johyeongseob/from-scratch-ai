from Tutorial2_step1 import n_letters, all_letters, all_categories, category_lines, n_categories
import random
import torch


# 임의의 category 및 그 category에서 무작위 줄(이름) 얻기
def randomTrainingPair():
    category = random.choice(all_categories)
    line = random.choice(category_lines[category])
    return category, line

# Category를 위한 One-hot 벡터
def categoryTensor(category):
    category_index = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][category_index] = 1
    return tensor

# 입력을 위한 처음부터 마지막 문자(EOS 제외)까지의  One-hot 행렬
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for char_idx in range(len(line)):
        letter = line[char_idx]
        tensor[char_idx][0][all_letters.find(letter)] = 1
    return tensor

# 목표를 위한 두번째 문자 부터 마지막(EOS)까지의 ``LongTensor``
def targetTensor(line):
    letter_indexes = []
    for char_idx in range(1, len(line)):
        current_char = line[char_idx]
        char_position = all_letters.find(current_char)
        letter_indexes.append(char_position)

    # EOS (End of Sequence) 토큰 추가
    letter_indexes.append(n_letters - 1)

    return torch.LongTensor(letter_indexes)


def randomTrainingExample():
    category, line = randomTrainingPair()
    category_tensor = categoryTensor(category)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return category_tensor, input_line_tensor, target_line_tensor

if __name__ == "__main__":
    category_tensor, input_line_tensor, target_line_tensor = randomTrainingExample()
    print(f"Category_tensor: {category_tensor.size()}, Name_tensor: {input_line_tensor.size()}, "
          f"Target_tensor: {target_line_tensor}")
