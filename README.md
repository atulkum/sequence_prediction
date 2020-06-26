# Various BERT and BiLSTM based model for NER 

## TO-DO
### Datset
- - [x] conll2003
- - [ ] atis

### Neural NER
- - [x] CharLSTM+WordLSTM+CRF: [Lample .etc, NAACL16](http://www.aclweb.org/anthology/N/N16/N16-1030.pdf)
  - - [x] Make a CoNLL-2003 batcher
  - - [x] Implement trainer
  - - [x] Implement WordLSTM + softmax
  - - [x] Implement CharLSTM + WordLSTM + softmax
  - - [x] Implement CharLSTM + WordLSTM + CRF
- - [x] Tranformer encoder + CRF
- - [x] BERT encoder + CRF
- - [x] pytorch JIT compilable Viterbi Decoder  https://github.com/atulkum/sequence_prediction/blob/master/NER_BERT/decoder.py#L9

### Slot Filling + intent prediciton
- - [ ] [Attention-Based Recurrent Neural Network Models for Joint Intent Detection and Slot Filling](https://arxiv.org/abs/1609.01454)
  - - [ ] Make a ATIS batcher
  - - [ ] Implement trainer
  - - [ ] Implement slot filler
  - - [ ] Implement intent
  
### Tree VAE
- - [ ] [STRUCTVAE: Tree-structured Latent Variable Models for Semi-supervised Semantic Parsing](https://arxiv.org/abs/1806.07832)

## Requiremet (python 3)
```
conda install pytorch  -c pytorch

```
CoNLL-2003 can be downloaded from https://www.clips.uantwerpen.be/conll2003/ner/

ATIS dataset can be downloaded from [split 0](http://lisaweb.iro.umontreal.ca/transfert/lisa/users/mesnilgr/atis/atis.fold0.pkl.gz) [split 1](http://lisaweb.iro.umontreal.ca/transfert/lisa/users/mesnilgr/atis/atis.fold1.pkl.gz) [split 2](http://lisaweb.iro.umontreal.ca/transfert/lisa/users/mesnilgr/atis/atis.fold2.pkl.gz) [split 3](http://lisaweb.iro.umontreal.ca/transfert/lisa/users/mesnilgr/atis/atis.fold3.pkl.gz) [split 4](http://lisaweb.iro.umontreal.ca/transfert/lisa/users/mesnilgr/atis/atis.fold4.pkl.gz)
