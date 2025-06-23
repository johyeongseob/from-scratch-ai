import os
import torch
from Tutorial3_step2 import EncoderRNN, AttnDecoderRNN
from Tutorial3_step3 import get_dataloader
from Tutorial3_step4 import train, evaluateRandomly


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_size = 128
batch_size = 32
n_epochs = 80

# 가중치 파일 경로
encoder_path = "../weight/Tutorial3/encoder_weights.pth"
decoder_path = "../weight/Tutorial3/decoder_weights.pth"

input_lang, output_lang, train_dataloader = get_dataloader(batch_size)

encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

# 가중치 파일이 존재하는지 확인
if os.path.exists(encoder_path) and os.path.exists(decoder_path):
    # 모델 가중치 불러오기
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))

else:
    # 모델 훈련 모드 설정
    encoder.train()
    decoder.train()

    # 모델 훈련
    train(train_dataloader, encoder, decoder, n_epochs)

    # 모델 가중치 저장
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)

# 모델 테스트 모드 설정
encoder.eval()
decoder.eval()

# 평가 진행
evaluateRandomly(encoder, decoder)
