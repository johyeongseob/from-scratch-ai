from Tutorial1_step2 import n_letters
from Tutorial1_step3 import n_categories
from Tutorial1_step5 import RNN
from Tutorial1_step6 import categoryFromOutput, randomTrainingExample
import torch
import torch.nn as nn
import torch.optim as optim


iters = 100000

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)
criterion = nn.NLLLoss()
optimizer = optim.SGD(rnn.parameters(), lr=0.005)

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()
    output = rnn.initoutput()

    optimizer.zero_grad()
    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)

    loss.backward()
    optimizer.step()

    return output, loss.item()


if __name__ == "__main__":
    rnn.train()
    for itr in range(1, iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output, loss = train(category_tensor, line_tensor)

        # ``iter`` 숫자, 손실, 이름, 추측 화면 출력
        if itr % 5000 == 0:
            guess = categoryFromOutput(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print(f"Epoch: {itr}/{iters}, Loss: {loss:.4f}, Name: {line} / Guess: {guess} {correct}")

        if itr == iters:
            torch.save(rnn.state_dict(), '../weight/rnn_model_weight1.pth')
            print("\n모델 가중치가 저장되었습니다.")
