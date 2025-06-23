"""
이 RNN 모듈은 《기본(vanilla)적인 RNN》을 구현하며,
입력과 은닉 상태(hidden state), 그리고 출력 뒤 동작하는
LogSoftmax 계층이 있는 3개의 선형 계층만을 가집니다.
"""

import torch
import torch.nn as nn
from Tutorial1_step2 import n_letters
from Tutorial1_step3 import n_categories
from Tutorial1_step4 import letterToTensor, lineToTensor

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        hidden = torch.tanh(self.i2h(input) + self.h2h(hidden))
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

    def initoutput(self):
        return torch.zeros(1, self.output_size)

if __name__ == "__main__":
    n_hidden = 128
    rnn = RNN(n_letters, n_hidden, n_categories)

    # Example 1
    input = letterToTensor('A')
    hidden = torch.zeros(1, n_hidden)

    output, next_hidden = rnn(input, hidden)

    print(f"input: {input.size()}")
    print(f"output: {output.size()}, next_hidden: {next_hidden.size()}\n")

    # Example 2
    input = lineToTensor('Albert')
    hidden = rnn.initHidden()
    output = rnn.initoutput()

    for i in range(input.size()[0]):
        output, hidden = rnn(input[i], hidden)

    print(f"input: {input.size()}, output: {output.size()}, hidden: {hidden.size()}")
