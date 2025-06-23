import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # 입력 인덱스(input_indices)는 Embedding 가중치 행렬(input_size)의 특정 행(row)을 선택
        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=hidden_size)
        # hidden_size: 입력 크기와 출력 히든 상태의 크기 (같은 크기로 설정)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))  # [batch_size, seq_len, hidden_size]
        # output: 모든 타임 스텝의 히든 상태 [batch_size, seq_len, hidden_size]
        # hidden: 마지막 타임 스텝의 히든 상태 [num_layers, batch_size, hidden_size]
        outputs, hidden = self.gru(embedded)
        return outputs, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))  # query: h_{dec,t}, keys: h_{enc,i}
        scores = scores.squeeze(2).unsqueeze(1)  # Softmax 적용 값은 마지막 차원: [batch_size, 4, 1] -> [batch_size, 1, 4]

        weights = self.softmax(scores)
        context = torch.bmm(weights, keys)  # [B,N,M] x [B,M,P] = [B,N,P]

        return context, weights


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=output_size, embedding_dim=hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(input_size=2 * hidden_size, hidden_size=hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
        self.LogSoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        # 시작 토큰(Start of Sentence Token, SOS_token), [B,1] = [[1],[1],...,[1]]
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        # hidden: 마지막 타임스텝의 히든 상태 [num_layers, batch_size, hidden_size]
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing 포함: 목표를 다음 입력으로 전달
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # Teacher forcing 미포함: 자신의 예측을 다음 입력으로 사용
                _, top_idx = decoder_output.topk(1)
                decoder_input = top_idx.squeeze(-1).detach()  # 입력으로 사용할 부분을 히스토리에서 분리

        # [Batch_size, seq_len, vocab_size] -> vocab_size 에 대한 Softmax를 사용하여 추정 단어를 선택
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = self.LogSoftmax(decoder_outputs)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        # decoder_input: [batch_size, 1, hidden_size]
        embedded = self.dropout(self.embedding(input))

        # 디코더의 히든 상태: [batch_size, 1, hidden_size]
        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        # *목표 언어의 모든 단어에 대한 점수(logit)
        output = self.out(output)

        return output, hidden, attn_weights

if __name__ == '__main__':
    input_size = 10  # 입력 어휘 사전 크기 (예: 50개의 고유 단어)
    hidden_size = 128  # GRU 히든 상태 크기
    output_size = 10  # 출력 어휘 사전 크기 (예: 50개의 고유 단어)
    batch_size = 2  # 배치 크기
    seq_len = 10  # 입력 시퀀스 길이

    encoder = EncoderRNN(input_size=input_size, hidden_size=hidden_size).to(device)
    attention = BahdanauAttention(hidden_size)
    decoder = AttnDecoderRNN(hidden_size=hidden_size, output_size=output_size).to(device)

    input_tensor = torch.randint(0, input_size, (batch_size, seq_len), dtype=torch.long, device=device)
    target_tensor = torch.randint(0, output_size, (batch_size, seq_len), dtype=torch.long, device=device)
    keys = torch.randn(batch_size, seq_len, hidden_size)
    query = torch.randn(batch_size, 1, hidden_size)
    context, weights = attention(query, keys)

    encoder_outputs, encoder_hidden = encoder(input_tensor)
    decoder_outputs, decoder_hidden, _ = decoder(encoder_outputs, encoder_hidden, target_tensor=target_tensor)

    # Example 1
    print(f"Example 1: Input Tensor Shape: {input_tensor.shape}, Encoder Outputs Shape: {encoder_outputs.shape}, "
          f"Encoder Hidden Shape: {encoder_hidden.shape}\n")  # (1, 배치 크기, 히든 크기)

    # Example 2
    print(f"Example 2: Keys shape: {keys.shape}, Query shape: {query.shape}, Scores shape: {weights.shape}, "
          f"Context shape: {context.shape}\n")

    # Example 3
    print(f"Example 3: Decoder Outputs Shape: {decoder_outputs.shape}, Decoder Hidden Shape: {decoder_hidden.shape}")
