# -*- coding:utf-8 -*-
import json

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np


# Helper functions to make the code more readable.
def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
# 最后return处应用了计算技巧，目的是防止sum后数据过大越界，实际就是对vec应用log_sum_exp
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    # print('max_score is :',max_score)
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    # print(max_score_broadcast)
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

# Create model
class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim # 5
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size # 956个词
        self.tag_to_ix = tag_to_ix # {'date': 0, 'title': 1, 'h1': 2, 'h2': 3, 'h3': 4, 'content': 5, '<START>': 6, '<STOP>': 7}
        self.tagset_size = len(tag_to_ix) # 8
        # nn.Embedding(len,5) 词表长度为n,每一个词向量的维度为5
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
         # // 代表向下取整(5,2)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True) # 用到batch时候，加上 , batch_first=True

        # Maps the output of the LSTM into tag space.
        # 将LSTM的输出映射到标记空间。
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of transitioning *to* i *from* j.
        # 随机生成的转移矩阵，放入了网络中，会更新的
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size).cuda())
        self.START_TAG = "<START>"
        self.STOP_TAG = "<STOP>"
        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        # 这两个语句强制执行一个约束，即我们从不转移到开始标记，也从不从停止标记转移
        # transitions:
        # tensor([[-1.5256e+00, -7.5023e-01, -6.5398e-01, -1.6095e+00, -1.0000e+04],
        #         [-6.0919e-01, -9.7977e-01, -1.6091e+00, -7.1214e-01, -1.0000e+04],
        #         [ 1.7674e+00, -9.5362e-02,  1.3937e-01, -1.5785e+00, -1.0000e+04],
        #         [-1.0000e+04, -1.0000e+04, -1.0000e+04, -1.0000e+04, -1.0000e+04],
        #         [-5.6140e-02,  9.1070e-01, -1.3924e+00,  2.6891e+00, -1.0000e+04]])
        self.transitions.data[tag_to_ix[self.START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[self.STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self, batch_size=1):
        # 元组，(tensor([[[0.6614, 0.2669]],
        #         [[0.0617, 0.6213]]]),
        #  tensor([[[-0.4519, -0.1661]],
        #         [[-1.5228,  0.3817]]]))
        return (torch.randn(2, batch_size, self.hidden_dim // 2).cuda(),
                torch.randn(2, batch_size, self.hidden_dim // 2).cuda())

    # 此处基于前向算法，计算输入序列x所有可能的标注序列对应的log_sum_exp，同时可参考上述LSE的说明.
    # 预测序列的得分
    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        # 使用正向传播算法计算分割函数
        init_alphas = torch.full((1, self.tagset_size), -10000.).cuda()
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[self.START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        # forward_var初值为[-10000.,-10000.,-10000.,0,-10000.] 初始状态的forward_var，随着step t变化
        forward_var = init_alphas

        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep，alphas_t即是上述定义的LSE，其size=1*tag_size
            for next_tag in range(self.tagset_size): # 此处遍历step=t时所有可能的tag
                # broadcast the emission score: it is the same regardless of the previous tag

                # feat: tensor([-0.2095,  0.1737, -0.3876,  0.4378, -0.3475]), torch.Size([5])
                # tensor(-0.2095) -> tensor([[-0.2095]]) -> tensor([[-0.2095, -0.2095, -0.2095, -0.2095, -0.2095]])
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size).cuda() #维度是１＊５
                # the ith entry of trans_score is the score of transitioning to next_tag from i
                # 从每一个tag转移到next_tag对应的转移score
                # 0时为例, tensor([-1.5256e+00, -7.5023e-01, -6.5398e-01, -1.6095e+00, -1.0000e+04]) -> tensor([[-1.5256e+00, -7.5023e-01, -6.5398e-01, -1.6095e+00, -1.0000e+04]])
                trans_score = self.transitions[next_tag].view(1, -1).cuda() # 维度是１＊５

                # 第一次迭代时理解：
                # trans_score所有其他标签到Ｂ标签的概率
                # 由lstm运行进入隐层再到输出层得到标签Ｂ的概率，emit_score维度是１＊５，5个值是相同的
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the scores.
                # 当前tag的forward变量是对所有分数进行log-sum-exp
                alphas_t.append(log_sum_exp(next_tag_var).view(1))  # 此处相当于LSE(t,next_tag)
            # 不断更新forward_var, 得到第（t-1）step时５个标签的各自分数，得到一条(1,5)的tensor矩阵
            forward_var = torch.cat(alphas_t).view(1, -1) # size=1*tag_size
        # 最后再走一步，代表序列结束。
        # 和STOP_TAG那一行相加，仍得到一条(1,5)的tensor矩阵
        terminal_var = forward_var + self.transitions[self.tag_to_ix[self.STOP_TAG]]
        # 用log_sum_exp计算分数
        alpha = log_sum_exp(terminal_var)
        return alpha

    # 得到feats
    def _get_lstm_features(self, sentences, sentence_lens): # torch.Size([128, 100]) torch.Size([128, 1])
        sentence_feats = []
        for ix, (sentence, sentence_len)in enumerate(zip(sentences, sentence_lens)): # torch.Size([1, 100]) torch.Size([1, 1])
            self.hidden = self.init_hidden()
            # 截取真实的句子长度
            sentence = sentence[0:sentence_len]
            # x.view(a,1,-1),代表将x打平，然后平均分为a行
            embeds = self.word_embeds(sentence).view(len(sentence), 1, -1) # embeds shape: torch.Size([sentence_len, 1, embedding_dim])
            # lstm_out shape: (句长, 1, hidden_dim), lstm_out.shape[0]: 句长或单词个数
            lstm_out ,self.hidden = self.lstm(embeds, self.hidden)
            total_sentence_out = torch.zeros([1, self.hidden_dim]).cuda() # total_sentence_out shape: torch.Size([1, hidden_dim])
            for i, out in enumerate(lstm_out):
                total_sentence_out += out
            avg_sentence_out = total_sentence_out/lstm_out.shape[0] # avg_sentence_out shape: torch.Size([1, hidden_dim])
            sentence_out = self.hidden2tag(avg_sentence_out) # sentence_out shape: torch.Size([1, tag_size])
            sentence_feats.append(sentence_out)
        lstm_feats = torch.cat(sentence_feats) # lstm_feats shape: torch.Size([128, 8])
        # print(lstm_feats.shape)
        return lstm_feats

    # 得到gold_seq tag的score
    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        # socre: tensor([0.])
        score = torch.zeros(1).cuda()
        # 将START_TAG创建的tensor([3.])拼接到tags序列上
        # tensor([3.])与tensor([0, 1, 1, 1, 2, 2, 2, 0, 1, 2, 2])拼接起来 -> tensor([3, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2, 2])
        tags = torch.cat([torch.tensor([self.tag_to_ix[self.START_TAG]], dtype=torch.long).cuda(), tags.cuda()]).cuda()
        # 将feats[i][tags[i+1]], 即feats[i][tag[i]]+tran[tag[i],tag[i-1]]相加得到分数
        for i, feat in enumerate(feats):
            # self.transitions[tags[i + 1], tags[i]] 实际得到的是从标签i到标签i+1的转移概率
            # feat[tags[i+1]], feat是step i 的输出结果，有５个值，对应B, I, E, START_TAG, END_TAG, 取对应标签的值
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[self.STOP_TAG], tags[-1]]
        return score

    # 解码，得到预测的序列，以及预测序列的得分
    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        # 初始化日志空间中的viterbi变量，全为-10000.0   tensor([[-10000., -10000., -10000., -10000., -10000.]])
        init_vvars = torch.full((1, self.tagset_size), -10000.).cuda()
        # START_TAG对应的变为0, 此时为tensor([[-10000., -10000., -10000.,      0., -10000.]])
        init_vvars[0][self.tag_to_ix[self.START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        # 步骤i的forward_var保存步骤i-1的viterbi变量
        forward_var = init_vvars
        # 按行进行for循环，共循环len次, len为句长， 每次维度为(1,5)
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step，timestamp=t时每个tag对应的最优路径其前一步时的tag
            viterbivars_t = []  # holds the viterbi variables for this step，timestamp=t时每个tag对应的最大score(不含发射概率)

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # next_tag_var[i]保存上一步骤中标记i的viterbi变量，加上从标记i转换到下一个标记的分数。

                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                # 我们这里不包括排放分数，因为最大值不依赖于它们（我们在下面添加它们）

                # transitions:
                # tensor([[-1.5256e+00, -7.5023e-01, -6.5398e-01, -1.6095e+00, -1.0000e+04],
                #         [-6.0919e-01, -9.7977e-01, -1.6091e+00, -7.1214e-01, -1.0000e+04],
                #         [ 1.7674e+00, -9.5362e-02,  1.3937e-01, -1.5785e+00, -1.0000e+04],
                #         [-1.0000e+04, -1.0000e+04, -1.0000e+04, -1.0000e+04, -1.0000e+04],
                #         [-5.6140e-02,  9.1070e-01, -1.3924e+00,  2.6891e+00, -1.0000e+04]])
                # forward_var: tensor([[-10000., -10000., -10000.,      0., -10000.]])
                next_tag_var = forward_var + self.transitions[next_tag] #其他标签（B,I,E,Start,End）到标签next_tag的概率
                best_tag_id = argmax(next_tag_var)
                # 记录每行中最大值的索引
                bptrs_t.append(best_tag_id)
                # 记录每行的最大值的tensor值
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            # 此处两个1*tag_size的向量相加，这样就得到timestamp=t时每个tag对应的完整最大score
            # 从step0到step(i-1)时5个序列中每个序列的最大score
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            # bptrs_t有５个元素
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        # 其他标签到STOP_TAG的转移概率
        terminal_var = forward_var + self.transitions[self.tag_to_ix[self.STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        #从后向前走，找到一个best路径
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[self.START_TAG]  # Sanity check
        best_path.reverse() # 把从后向前的路径正过来
        # 返回路径得分，最优路径列表
        return path_score, best_path

    def neg_log_likelihood(self, sentences, tags, sentence_lens):
        # 将lstm输出，经过linear层，得到feats: (句长,5)维度的矩阵
        feats = self._get_lstm_features(sentences, sentence_lens).cuda()
        forward_score = self._forward_alg(feats)
        # print(feats.shape, tags.view(-1).shape) # tags shape: torch.Size([128, 1])
        gold_score = self._score_sentence(feats, tags.view(-1)) # torch.Size([128, 8]) torch.Size([128]), gold_score是数字
        return forward_score - gold_score


    def forward(self, sentences, sentence_lens):  # dont confuse this with _forward_alg above.
        # print("用到forward了！！！")
        # Get the emission scores from the BiLSTM
        # 从BiLSTM得到排放分数，将LSTM的输出映射到标记空间, lstm_feats为4维到5维，即维度为(句长，5)
        lstm_feats = self._get_lstm_features(sentences, sentence_lens)

        # Find the best path, given the features.
        # 找到最佳路径，给定特征。得到最优得分和标签路径列表
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
    # Helper functions to make the code more readable.
    def argmax(self,vec):
        # return the argmax as a python int
        _, idx = torch.max(vec, 1)
        return idx.item()

    def prepare_sequence(self, seq, to_ix):
        idxs = [to_ix[w] for w in seq]
        return torch.tensor(idxs, dtype=torch.long)

    # Compute log sum exp in a numerically stable way for the forward algorithm
    # 最后return处应用了计算技巧，目的是防止sum后数据过大越界，实际就是对vec应用log_sum_exp
    def log_sum_exp(self, vec):
        max_score = vec[0, self.argmax(vec)]
        # print('max_score is :',max_score)
        max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
        # print(max_score_broadcast)
        return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
