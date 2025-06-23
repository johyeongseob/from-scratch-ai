"""
학습 데이터 준비
"""

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from Tutorial3_step1 import *
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10

def indexesFromSentence(lang, sentence):
    indexes = []
    for word in sentence.split(' '):
        indexes.append(lang.word2index[word])
    return indexes


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)  # size: [1, n], n: num of words


def get_dataloader(batch_size):
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)

    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)  # (11445, 10)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)  # (11445, 10)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids   # e.g. [ 116  116 3272  961    1    0    0    0    0    0]
        target_ids[idx, :len(tgt_ids)] = tgt_ids  # e.g. [  2    3   146   32    2    3   65    1    0    0]

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader

if __name__ == '__main__':
    input_lang, output_lang, train_dataloader = get_dataloader(batch_size=1)

    for i, (input_ids, target_ids) in enumerate(train_dataloader):
        print(f"Input Ids (영어): {input_ids}")
        print(f"Target Ids (프랑스어): {target_ids}")
        break
