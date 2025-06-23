# register buffer in Pytorch ->
# If you have parameters in your model, which should be saved and restored in the state_dict,
# but not trained by the optimizer, you should register them as buffers.

import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, embed_model_dim):
        """
        Args:
            seq_len: length of input sequence
            embed_model_dim: demension of embedding
        """
        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_model_dim

        # Positional Encoding 계산
        pe = torch.zeros(max_seq_len, self.embed_dim)
        for pos in range(max_seq_len):
            for i in range(0, self.embed_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / self.embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / self.embed_dim)))
        pe = pe.unsqueeze(0)  # Batch dimension 추가
        self.register_buffer('pe', pe)  # 학습 불가능한 텐서로 등록

    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            x: output
        """

        # make embeddings relatively larger
        x = x * math.sqrt(self.embed_dim)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x

if __name__ == '__main__':
    max_seq_len = 50  # 최대 시퀀스 길이
    embed_model_dim = 64  # 임베딩 차원

    pos_embedding = PositionalEmbedding(max_seq_len=max_seq_len, embed_model_dim=embed_model_dim)

    # 입력 데이터 (배치 크기 2, 시퀀스 길이 10, 임베딩 차원 64)
    x = torch.randn(2, 10, embed_model_dim)  # (batch_size, seq_len, embed_dim)

    output = pos_embedding(x)

    # 출력 크기 확인
    print("Input shape: ", x.shape)  # Input shape: (2, 10, 64)
    print("Output shape: ", output.shape)  # Output shape: (2, 10, 64)

    # Positional Encoding 값만 확인
    pos_encoding = pos_embedding.pe[0, :5, :5]
    print("Positional Encoding (first 5 positions, first 5 dimensions):")
    print(pos_encoding)
