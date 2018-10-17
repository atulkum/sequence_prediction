# Neural NER

## TO-DO
- - [ ] CharLSTM+WordLSTM+CRF: [Lample .etc, NAACL16](http://www.aclweb.org/anthology/N/N16/N16-1030.pdf)
  - - [x] Make a CoNLL-2003 batcher using pytorchtext
  - - [ ] Implement trainer
  - - [ ] Implement WordLSTM + softmax
  - - [ ] Implement WordLSTM + CRF
  - - [ ] Implement CharLSTM + WordLSTM + CRF

## Requiremet (python 3)
```
conda install pytorch  -c pytorch
pip install torchtext

```
Conll2003 data can be downloaded from https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003
