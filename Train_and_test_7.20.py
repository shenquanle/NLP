# -*- coding:utf-8 -*-
import json

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.utils.data as Data

from BiLSTM_CRF import BiLSTM_CRF
from YuanDataset import YuanDataset



# Run training
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 50
HIDDEN_DIM = 100
BATCH_SIZE = 128
EPOCH = 10
BATCH_NUM = 148

vocab_path = "./jieba_vocab.txt"
train_corpus_path = "./jieba_corpus2.txt"
train_label_path = "./raw_label.txt"

test_corpus_path = "./test1_jieba.txt"
test_label_path = "./test1_label.txt"

tag_to_ix = {"date": 0, "title": 1, "h1": 2, "h2": 3, "h3": 4, "content": 5, "<START>": 6, "<STOP>": 7}
yuan = YuanDataset(vocab_path, train_corpus_path, train_label_path, test_corpus_path, test_label_path)
train_dataset, test_dataset, word_to_ix = yuan.prepare_dataset()
# sentence, label, sentecce_len
train_loader = yuan.get_dataloader(train_dataset, BATCH_SIZE)
test_loader = yuan.get_dataloader(test_dataset, BATCH_SIZE)

# for batch in test_loader:
#     print(len(batch[0]),batch[0])

# (词典长度，标签字典，embedding 维度， 隐藏层数)
model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM).cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
print(model)
print(optimizer)


print('='*10+'predictions before training'+'='*10)
# Check predictions before training
with torch.no_grad():
    for ix, batch in enumerate(test_loader):
        print(ix, batch[0].shape, batch[2].view(BATCH_SIZE, -1).shape)
        print(model(batch[0].cuda(), batch[2].view(BATCH_SIZE, -1).cuda()))
print('='*10+'predictions before training done!!!'+'='*10)
# # 合
for epoch in range(EPOCH):  # again, normally you would NOT do 300 epochs, it is toy data
    i = 0
    for batch in train_loader:
        model.train()
        model.zero_grad()
        loss = model.neg_log_likelihood(batch[0].cuda(), batch[1].cuda(), batch[2].view(BATCH_SIZE, -1).cuda())
        loss.backward()
        optimizer.step()
        i+=1
        print("|Train Epoch {:2d}/{:2d} | batch {:2d}/{:2d} Loss {:5.3f} |".format(epoch, EPOCH, i, BATCH_NUM, loss.item()))

torch.save(model.state_dict(), './yuan_model_10.pkl')

# state_dict = torch.load('./yuan_model_10.pkl')
# model.load_state_dict(state_dict)

# Check predictions after training
print('='*10+'predictions after training'+'='*10)
result = []
model.eval()
with torch.no_grad():
    for batch in test_loader:
        print(batch[0].shape)
        score ,label_dig = model(batch[0].cuda(), batch[2].view(BATCH_SIZE, -1).cuda())
        result.append(label_dig)
        print(score, label_dig)
print(result)
print('='*10+'predictions after training done!!!'+'='*10)
