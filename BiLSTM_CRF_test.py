# -*- coding:utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(1)

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
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        # nn.Embedding(17,5) 词表长度为17,每一个词向量的维度为5
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
         # // 代表向下取整
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        # 将LSTM的输出映射到标记空间。
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of transitioning *to* i *from* j.
        # 随机生成的转移矩阵，放入了网络中，会更新的
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        # 这两个语句强制执行一个约束，即我们从不转移到开始标记，也从不从停止标记转移
        # transitions:
        # tensor([[-1.5256e+00, -7.5023e-01, -6.5398e-01, -1.6095e+00, -1.0000e+04],
        #         [-6.0919e-01, -9.7977e-01, -1.6091e+00, -7.1214e-01, -1.0000e+04],
        #         [ 1.7674e+00, -9.5362e-02,  1.3937e-01, -1.5785e+00, -1.0000e+04],
        #         [-1.0000e+04, -1.0000e+04, -1.0000e+04, -1.0000e+04, -1.0000e+04],
        #         [-5.6140e-02,  9.1070e-01, -1.3924e+00,  2.6891e+00, -1.0000e+04]])
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        # 元组，(tensor([[[0.6614, 0.2669]],
        #         [[0.0617, 0.6213]]]),
        #  tensor([[[-0.4519, -0.1661]],
        #         [[-1.5228,  0.3817]]]))
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    # 此处基于前向算法，计算输入序列x所有可能的标注序列对应的log_sum_exp，同时可参考上述LSE的说明.
    # 预测序列的得分
    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        # 使用正向传播算法计算分割函数
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        # forward_var初值为[-10000.,-10000.,-10000.,0,-10000.] 初始状态的forward_var，随着step t变化
        forward_var = init_alphas

        # Iterate through the sentence，如下遍历一个句子的各个word或者step
        # (11,5),11为句长,5为词向量维度
        # tensor([[-0.2095,  0.1737, -0.3876,  0.4378, -0.3475],
        #         [-0.2681,  0.1620, -0.4196,  0.4297, -0.2857],
        #         [-0.3868,  0.2700, -0.4559,  0.3874, -0.2614],
        #         [-0.3761,  0.2536, -0.3897,  0.4786, -0.2404],
        #         [-0.3446,  0.1833, -0.4204,  0.4936, -0.0980],
        #         [-0.2738,  0.2778, -0.3540,  0.4534, -0.3920],
        #         [-0.2207,  0.2085, -0.4019,  0.3099, -0.6957],
        #         [-0.3363,  0.2813, -0.4552,  0.3353, -0.3985],
        #         [-0.3904,  0.0843, -0.5000,  0.3937, -0.2078],
        #         [-0.2801,  0.2033, -0.4282,  0.4708, -0.0854],
        #         [-0.2504,  0.3018, -0.3046,  0.4671, -0.5199]])
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep，alphas_t即是上述定义的LSE，其size=1*tag_size
            for next_tag in range(self.tagset_size): # 此处遍历step=t时所有可能的tag
                # broadcast the emission score: it is the same regardless of the previous tag

                # feat: tensor([-0.2095,  0.1737, -0.3876,  0.4378, -0.3475]), torch.Size([5])
                # tensor(-0.2095) -> tensor([[-0.2095]]) -> tensor([[-0.2095, -0.2095, -0.2095, -0.2095, -0.2095]])
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size) #维度是１＊５
                # the ith entry of trans_score is the score of transitioning to next_tag from i
                # 从每一个tag转移到next_tag对应的转移score
                # 0时为例, tensor([-1.5256e+00, -7.5023e-01, -6.5398e-01, -1.6095e+00, -1.0000e+04]) -> tensor([[-1.5256e+00, -7.5023e-01, -6.5398e-01, -1.6095e+00, -1.0000e+04]])
                trans_score = self.transitions[next_tag].view(1, -1) # 维度是１＊５

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
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        # 用log_sum_exp计算分数
        alpha = log_sum_exp(terminal_var)
        return alpha

    # 得到feats
    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        # x.view(a,1,-1),代表将x打平，然后平均分为a行
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1) # (句长, 1, 5)

        # lstm(5,4), lstm_out 维度为(句长,1,4), 4为lstm的隐藏层数
        # lstm_out,self.hidden = self.lstm(embeds, self.hidden) # (句长, 1, 4),
        lstm_out ,(h_n,c_n) = self.lstm(embeds,self.hidden)
        # out = torch.cat(h_n[-1, :, :], h_n[-2, :, :], dim=-1)

        # print('lstm_out is:',lstm_out)
        # print('out is:',h_n)
        # print('h_n_0 is:',h_n[0])
        # print('h_n_1 is:', h_n[1])
        # print('cat is :', torch.cat([h_n[0], h_n[1]],dim=-1))

        # 将lstm的输出tensor给reshape一下(句长，4)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)

        # 将LSTM的输出映射到标记空间, lstm_feats为4维变为5维，即维度为(句长，5)
        lstm_feats = self.hidden2tag(lstm_out)
        # print('lstm->linear:',lstm_feats,lstm_feats.shape)
        return lstm_feats

    # 得到gold_seq tag的score
    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        # socre: tensor([0.])
        score = torch.zeros(1)
        # 将START_TAG创建的tensor([3.])拼接到tags序列上
        # tensor([3.])与tensor([0, 1, 1, 1, 2, 2, 2, 0, 1, 2, 2])拼接起来 -> tensor([3, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2, 2])
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        # 将feats[i][tags[i+1]], 即feats[i][tag[i]]+tran[tag[i],tag[i-1]]相加得到分数
        for i, feat in enumerate(feats):
            # self.transitions[tags[i + 1], tags[i]] 实际得到的是从标签i到标签i+1的转移概率
            # feat[tags[i+1]], feat是step i 的输出结果，有５个值，对应B, I, E, START_TAG, END_TAG, 取对应标签的值
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    # 解码，得到预测的序列，以及预测序列的得分
    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        # 初始化日志空间中的viterbi变量，全为-10000.0   tensor([[-10000., -10000., -10000., -10000., -10000.]])
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        # START_TAG对应的变为0, 此时为tensor([[-10000., -10000., -10000.,      0., -10000.]])
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

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
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
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
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse() # 把从后向前的路径正过来
        # 返回路径得分，最优路径列表
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        # 将lstm输出，经过linear层，得到feats: (句长,5)维度的矩阵
        feats = self._get_lstm_features(sentence)
        print("tags is:",tags)
        # 传入feats, 算出句子前向传播的得分
        forward_score = self._forward_alg(feats)
        # 传入feats和tags,
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        print("用到forward了！！！")
        # Get the emission scores from the BiLSTM
        # 从BiLSTM得到排放分数，将LSTM的输出映射到标记空间, lstm_feats为4维到5维，即维度为(句长，5)
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        # 找到最佳路径，给定特征。得到最优得分和标签路径列表
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


# Run training
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

# Make up some training data
training_data = [("the wall street journal reported today that apple corporation made money".split(),"B I I I O O O B I O O".split()),
                 ("georgia tech is a university in georgia".split(),"B I O O O O B".split())]
test = [("John lives in New York and works for the European Union".split(),"B O O B I O O O O B I".split())]

# 获取词典及其长度
word_to_ix = {}
i = -1
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            i += 1
            word_to_ix[word] = i
j = 0
for sentence, tags in test:
    for word in sentence:
        if word not in word_to_ix:
            i += 1
            word_to_ix[word] = j+i
tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

# test = torch.tensor([[-1.6441e+00, -1.4994e+00,  2.2446e+00, -1.8893e-01,  2.4370e-01],
#         [-1.9552e+00, -1.0077e+00,  1.9722e+00, -1.3767e-01,  1.3255e-01],
#         [-1.5157e+00, -1.4592e+00,  2.1405e+00, -1.8739e-01,  2.3376e-01],
#         [ 1.4305e+00, -1.5022e+00, -4.9542e-01, -2.8754e-01,  6.8662e-01],
#         [-1.8601e+00, -6.8110e-04,  6.4632e-01, -1.3445e-01,  1.4648e-01],
#         [-1.2051e+00, -1.2172e+00,  1.6206e+00, -1.8814e-01,  2.4905e-01],
#         [-9.6636e-01, -1.3575e+00,  1.4808e+00, -2.1664e-01,  3.4384e-01],
#         [-2.0748e+00,  3.1798e-01,  7.5417e-01, -7.1617e-02, -6.2180e-02],
#         [-1.0955e+00, -8.3888e-01,  1.3184e+00, -1.4194e-01,  1.1228e-01],
#         [-1.9791e+00,  1.0666e+00,  7.4876e-03,  4.5318e-03, -2.3343e-01],
#         [-2.1014e+00,  4.3115e-01,  7.1764e-01, -2.7786e-02, -1.4580e-01]])
# for ix ,i in enumerate(test):
#     print(ix,i, nn.functional.softmax(i))

# (词典长度，标签字典，embedding 维度， 隐藏层数)
model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)

optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
print(model)
print(optimizer)
print('='*10+'predictions before training'+'='*10)
# Check predictions before training
with torch.no_grad():
    # 句子中单词对应词典的索引列表
    precheck_sent = prepare_sequence(test[0][0], word_to_ix)
    print(precheck_sent)
    # 句子中单词对应标签的索引列表
    # precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    # 得分: (tensor(2.6907), 路径列表: [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1])
    print(model(precheck_sent))
print('='*10+'predictions before training done!!!'+'='*10)
# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(10):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        model.train()
        # Step 1. Remember that Pytorch accumulates gradients. 记住pytorch累计梯度
        # We need to clear them out before each instance 我们需要在每一个实例发生前把它们清除掉
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices. 为网络做好准备，让我们的输入的句子转换成单词索引的时态。
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # Step 3. Run our forward pass.
        print(sentence)
        loss = model.neg_log_likelihood(sentence_in, targets)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()

# Check predictions after training
print('='*10+'predictions after training'+'='*10)
with torch.no_grad():
    model.eval()
    precheck_sent = prepare_sequence(test[0][0], word_to_ix)
    score, label_dig = model(precheck_sent)
    print(model(precheck_sent))
    real_lab = [tag_to_ix[i] for i in test[0][1]]
    print(label_dig)
    print(real_lab)
    acc = 0
    for i, j in zip(real_lab, label_dig):
        if i == j:
            acc +=1
    print('accrancy is :',acc/len(real_lab))
print('='*10+'predictions after training done!!!'+'='*10)
# We got it!