from Tutorial2_step1 import n_letters
from Tutorial2_step2 import RNN
from Tutorial2_step3 import randomTrainingExample

import torch
import torch.nn as nn
import torch.optim as optim


iters = 100000
n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_letters)

criterion = nn.NLLLoss()
optimizer = optim.SGD(rnn.parameters(), lr=0.005)

def train(category_tensor, input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    hidden = rnn.initHidden()

    optimizer.zero_grad()
    rnn.zero_grad()

    loss = torch.Tensor([0])

    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()
    optimizer.step()

    return loss.item() / input_line_tensor.size(0)

if __name__ == "__main__":
    rnn.train()
    for itr in range(1, iters + 1):
        loss = train(*randomTrainingExample())

        if itr % 5000 == 0:
            print(f"Epoch: {itr}/{iters}, Loss: {loss:.4f}")

        if itr == iters:
            torch.save(rnn.state_dict(), '../weight/rnn_model_weight2.pth')
            print("\n모델 가중치가 저장되었습니다.")
