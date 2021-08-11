# NLP
my NLP experiments...
在NLP方面一共做了两个实验，都是基于Pytorch框架的，一个是NER的，一个是段落分类，NER实验思路和公文段落分类这个很像，就不放出来了。
由于保密性质，数据集不能公开。
## BiLSTM_CRF_test
利用Pytorch官网的Tutorials的源码示例改变的测试版代码
## BiLSTM_CRF
BiLSTM_CRF.py文件是定义的模型文件，自定义了init函数，forward函数；
YuanDataset是用来构建政府公文语料数据集，划分训练集和测试集、分batch构建dataloader的数据构建文件；
Train_and_test.py文件是训练文件，调用本地带标签的数据集，进行有监督的训练。
