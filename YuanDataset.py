# -*- coding:utf-8 -*-

import torch
import numpy as np
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data as Data


MAX_LEN = 100

def trunc_pad(_list: list, max_len=MAX_LEN, if_sentence=True):  # 设置输入句子的最大长度
    if len(_list) == max_len:
        return _list
    if len(_list) > max_len:
        if if_sentence:
            return _list[0:max_len]
    else:
        if if_sentence:
            _list.extend(["[PAD]"] * (max_len - len(_list)))
            return _list

class YuanDataset(Data.Dataset):
    def __init__(self, vocab_path, train_corpus_path, train_label_path, test_corpus_path, test_label_path):
        super(YuanDataset, self).__init__()
        self.vocab_path  = vocab_path
        self.train_corpus_path = train_corpus_path
        self.train_label_path = train_label_path
        self.test_corpus_path = test_corpus_path
        self.test_label_path = test_label_path

    def prepare_sequence(self, seq, word_to_ix):
        idxs = [word_to_ix[w] for w in seq]
        # return torch.tensor(idxs, dtype=torch.long)
        return idxs


    def get_vocab_dict(self, train_sentence_list, test_sentence_list):
        word_to_ix = {"[PAD]": 0}

        for sentence in train_sentence_list:
            for word in sentence:
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)
        # print(len(word_to_ix))
        for sentence in test_sentence_list:
            for word in sentence:
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)
        # print(len(word_to_ix))
        return word_to_ix

    def prepare_dataset(self):
        tag_to_ix = {"date": 0, "title": 1, "h1": 2, "h2": 3, "h3": 4, "content": 5, "<START>": 6, "<STOP>": 7}

        # get train sentence_list, label_list
        train_sentence_list = []
        train_label_list = []
        train_sentence_len_list = []
        with open(self.train_corpus_path, 'r', encoding='utf-8') as f:
            with open(self.train_label_path, 'r', encoding='utf-8') as ff:
                lines = f.readlines()
                labels = ff.readlines()
                idx = 0
                for line, label in zip(lines, labels):
                    idx += 1
                    line = line.replace('\ue010', '').replace('\n', '').split(' ')
                    label = [label.replace('\n', '')]
                    train_sentence_len_list.append(MAX_LEN if len(line) >= MAX_LEN else len(line))
                    train_sentence_list.append(line)
                    train_label_list.append(label)

        print(len(train_sentence_list))
        print(len(train_sentence_len_list))
        print(len(train_label_list))

        # get train sentence_list, label_list
        test_sentence_list = []
        test_label_list = []
        test_sentence_len_list = []
        with open(self.test_corpus_path, 'r', encoding='utf-8') as f:
            line_list = f.readlines()
            with open(self.test_label_path, 'r', encoding='utf-8') as ff:
                label_list = ff.readlines()

            for line, label in zip(line_list, label_list):
                line = line.replace('\ufeff', '').replace('\ue010', '').replace('\n', '').split(' ')
                label = label.replace('\ufeff', '').replace('\ue010', '').replace('\n', '').split(' ')
                test_sentence_list.append(line)
                test_sentence_len_list.append(MAX_LEN if len(line) >= MAX_LEN else len(line))
                test_label_list.append(label)
        print(len(test_sentence_list))
        print(len(test_sentence_len_list))
        print(len(test_label_list))
        # cut and padding
        for train_sentence_ix in range(len(train_sentence_list)):
            train_sentence_list[train_sentence_ix] = trunc_pad(train_sentence_list[train_sentence_ix], if_sentence=True)
            # print(train_sentence_list[train_sentence_ix])
            # if len(sentence_list[sentence_ix]) != 100:
            #     print(len(sentence_list[sentence_ix]),sentence_ix, sentence_list[sentence_ix])
        for test_sentence_ix in range(len(test_sentence_list)):
            test_sentence_list[test_sentence_ix] = trunc_pad(test_sentence_list[test_sentence_ix], if_sentence=True)
            # print(test_sentence_list[test_sentence_ix])

        # get all word's word_to_ix dict
        word_to_ix = self.get_vocab_dict(train_sentence_list,test_sentence_list)

        # turn sentence's word to ix and label to ix
        for i in range(len(train_sentence_list)):
            train_sentence_list[i] = [word_to_ix[t] for t in train_sentence_list[i]]
            train_label_list[i] = [tag_to_ix[t] for t in train_label_list[i]]

        for i in range(len(test_sentence_list)):
            test_sentence_list[i] = [word_to_ix[t] for t in test_sentence_list[i]]
            test_label_list[i] = [tag_to_ix[t] for t in test_label_list[i]]
        # 先转换成 torch 能识别的 Dataset
        train_dataset = Data.TensorDataset(torch.tensor(train_sentence_list, dtype=torch.long).cuda(),
                                           torch.tensor(train_label_list, dtype=torch.long).cuda(),
                                           torch.tensor(train_sentence_len_list, dtype=torch.long).cuda())
        test_dataset = Data.TensorDataset(torch.tensor(test_sentence_list, dtype=torch.long).cuda(),
                                          torch.tensor(test_label_list, dtype=torch.long).cuda(),
                                          torch.tensor(test_sentence_len_list, dtype=torch.long).cuda())
        return train_dataset, test_dataset, word_to_ix


    def get_dataloader(self, dataset, batch_size):
        data_loader = Data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False) # shuffle 决定了每次的batch是不是随机的 Trues:随机 /False: 不随机，固定次序
        print("dataloader done!")
        return data_loader

# vocab_path = "./jieba_vocab.txt"
# train_corpus_path = "./jieba_corpus2.txt"
# train_label_path = "./raw_label.txt"
# test_corpus_path = "./test1_jieba.txt"
# test_label_path = "./test1_label.txt"
# yuan = YuanDataset(vocab_path, train_corpus_path, train_label_path, test_corpus_path, test_label_path)
# train_dataset, test_dataset, word_to_ix = yuan.prepare_dataset()
# train_loader = yuan.get_dataloader(train_dataset, 128)
#
# for ix,batch in enumerate(train_loader):
#     if ix % 128 == 0:
#         print(ix,batch[0],batch[1],batch[2].view(128, -1))
# test = torch.tensor([[1,2,3]])
# b = torch.tensor([[1,2,3]])
# print(test.shape)
# print(test+b)