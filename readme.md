# Sentiment Analysis with Pytorch [WIP]

Pytorch Example For Aspect-based Sentiment Analysis with RNN / GRUs / LSTMs on SemEval 2014. Heavily modified from PTB example in official example repository.

# Usage

```
python train.py --cuda --batch-size 20 --rnn_type GRU
```

For some reason, LSTM not working yet but GRU and RNNs are working well. 

<!-- 
# Data Preperation

1) Download glove embeddings from paper. 

```
Jeffrey Pennington, Richard Socher, and Christopher D
Manning. 2014. Glove: Global vectors for word representation.
Proceedings of the Empiricial Methods
in Natural Language Processing (EMNLP 2014),
12:1532–1543.
```

2) Make directory
```
mkdir ../glove_embeddings
```

3) place glove.840B.300d.txt into ../glove_embeddings -->

