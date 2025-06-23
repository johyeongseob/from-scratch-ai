from Tutorial2_step1 import n_letters, all_letters
from Tutorial2_step2 import RNN
from Tutorial2_step3 import categoryTensor, inputTensor
import torch


n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_letters)
rnn.load_state_dict(torch.load('../weight/rnn_model_weight2.pth'))

max_length = 20

# 카테고리와 시작 문자로부터 샘플링 하기

def sample(category, start_letter='A'):
    with torch.no_grad():  # 샘플링에서 히스토리를 추적할 필요 없음
        category_tensor = categoryTensor(category)
        input = inputTensor(start_letter)
        hidden = rnn.initHidden()

        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn(category_tensor, input[0], hidden)
            _, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)

        return output_name


if __name__ == '__main__':
    # 하나의 카테고리와 여러 시작 문자들로 여러 개의 샘플 얻기
    def samples(category, start_letters='ABC'):
        for start_letter in start_letters:
            print(f"Predict name with {start_letter}: {sample(category, start_letter)}")

    if __name__ == '__main__':
        samples('Korean', 'KLP')
        # 예상 결과: 김(Kim), 이(Lee), 박(Park)
