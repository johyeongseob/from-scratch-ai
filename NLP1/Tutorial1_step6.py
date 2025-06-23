from Tutorial1_step2 import n_letters
from Tutorial1_step3 import all_categories, n_categories, category_lines
from Tutorial1_step4 import lineToTensor
from Tutorial1_step5 import RNN
import torch
import random


# RNN 예시 및 훈련 데이터 구성
def categoryFromOutput(output):
    _, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i]

def randomTrainingExample():
    category = random.choice(all_categories)
    line = random.choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

if __name__ == "__main__":

    # Example 1
    input = lineToTensor('Albert')
    n_hidden = 128

    rnn = RNN(n_letters, n_hidden, n_categories)
    hidden = rnn.initHidden()
    output = rnn.initoutput()

    for i in range(input.size()[0]):
        output, hidden = rnn(input[i], hidden)

    print(f"Albert's predict category: {categoryFromOutput(output)}\n")

    # Example 2
    print(f"10 random Training samples\n")
    for i in range(1, 11):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        print(f"Name: {line}, Category: {category}")
