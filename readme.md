# Sentiment Analysis with Pytorch [WIP]

Pytorch Example For Aspect-based Sentiment Analysis with RNN / GRUs / LSTMs on SemEval 2014. Heavily modified from PTB example in official example repository.

Currently we only implemented a baseline LSTM/RNN/GRU model with a linear layer on the *last* output. The sequences are padded with zeros *from the front* so that the last vector is not zero. We pad these in the prepare script using keras pad sequences. Nothing is masked so far and we pad to the max length. 

There are two modes of prediction, namely `term` and `aspect`. Aspect refers to aspect **categories** while term refers to, well, **terms** which are sequences that **can** be found in the text itself. There are two datasets, Laptop and Restaurants. There are both term and aspect settings for Laptop but only aspect setting for restaurants.

# Usage

Before running prepare.py you need a folder `../embedding/` one directory higher than project root. (That is where my glove embeddings to avoid copying them in every single project)

## Data Preperation

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

3) place glove.840B.300d.txt into ../glove_embeddings

```
python prepare.py         # Will take awhile.
```
This should build into `./store` and `./embeddings/`. The former is the environment object that `train.py` reads while the file written into embeddings is just a smaller concised version of glove so  that I can rerun prepare.py fast. 

```
python train.py --cuda --batch-size 20 --rnn_type GRU
```



# Notes

1) It works now! Testing on SemEval (Aspect Category + Restaurants) give about 77-79% accuracy around epoch 20. This is the same result I previously got using TensorFlow. The algorithm constantly predicts the same class (2) for the first 10+ iterations though. 

2) I can't find a way to use clip_norm with optimizers. So it's either you manually clip or you use the optimizer. (For now, I don't clip)

3) LSTM, RNN, GRU are all supported. 

4) It seems like RNNs in pyTorch are batch-minor, i.e, seq length is dim 0 and batch is dim 1. Let's wait for more variable length support.

5) Pretrained embeddings are supported. (I loaded GloVe)


# Additional Notes on Dataset Pre-processing

1) I fixed 3 aspect terms in the restaurant dataset which are **clearly** typos.

```
====================
[384, 4494, 358, 2389, 2569, 964, 4686, 182, 2817, 2580, 322, 3501, 303, 3387, 4065, 940, 2]
['good', 'atmosphere', 'combination', 'of', 'all', 'the', 'hottest', 'music', 'dress', 'code', 'is', 'relatively', 'strict', 'except', 'on', 'fridays', '']
['dress', 'cod']
[Warning] Target not found in text!
====================
[1632, 5071, 4437, 1065, 2865, 1419, 3900, 1380, 903, 119, 2930, 1844, 3201, 4002, 2]
['interesting', 'other', 'dishes', 'for', 'a', 'change', 'include', 'chicken', 'in', 'curry', 'sauce', 'and', 'salmon', 'caserole', '']
['chicken', 'in', 'curry', 'sauc']
[Warning] Target not found in text!
====================
[368, 4523, 4426, 3782, 897, 2044, 2465, 1496, 4558, 384, 4997, 2865, 3860, 4264, 4044, 2671, 964, 1674, 719, 2]
['they', 'should', 'have', 'called', 'it', 'mascarpone', 'with', 'chocolate', 'chips', 'good', 'but', 'a', 'far', 'cry', 'from', 'what', 'the', 'name', 'implies', '']
['mascarpone', 'with', 'chocolate', 'chip']
[Warning] Target not found in text!
```






