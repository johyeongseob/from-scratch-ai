from Tutorial1_step2 import n_letters
from Tutorial1_step3 import all_categories, n_categories
from Tutorial1_step4 import lineToTensor
from Tutorial1_step5 import RNN
import torch


n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)
rnn.load_state_dict(torch.load('../weight/rnn_model_weight1.pth'))

# 주어진 라인의 출력 반환
def evaluate(line_tensor):
    hidden = rnn.initHidden()
    output = rnn.initoutput()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

rnn.eval()
def predict(input_line, n_predictions=3):
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))

        # Get top N categories
        _, topi = output.topk(n_predictions, 1, True)

        predictions = []
        for i in range(n_predictions):
            category_index = topi[0][i].item()
            print(f"{input_line}'s {i+1}th predict category is {all_categories[category_index]}")

if __name__ == '__main__':
    predict('Dovesky')
    print("")
    predict('Jackson')
    print("")
    predict('Satoshi')
