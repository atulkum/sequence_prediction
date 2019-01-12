import codecs
from .sentence_utils import prepare_sentence
from .tag_scheme_utils import update_tag_scheme

def load_sentences(path, tag_scheme):
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf8'):
        line = line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)

    update_tag_scheme(sentences, tag_scheme)

    return sentences


def prepare_dataset(sentences, vocab, config):
    data = []
    for s in sentences:
        datum = prepare_sentence(s, vocab, config)
        tags = [vocab.tag_to_id[w[-1]] for w in s]
        datum['tags'] = tags

        data.append(datum)
    return data

''' 
import numpy as np
def insert_singletons(words, singletons, p=0.5):
    new_words = []
    for word in words:
        if word in singletons and np.random.uniform() < p:
            new_words.append(Constants.UNK_ID)
        else:
            new_words.append(word)
    return new_words

import re
import codecs
import os
def augment_with_pretrained(dictionary, ext_emb_path, words):
    print 'Loading pretrained embeddings from %s...' % ext_emb_path
    assert os.path.isfile(ext_emb_path)

    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(ext_emb_path) > 0
    ])

    if words is None:
        for word in pretrained:
            if word not in dictionary:
                dictionary[word] = 0
    else:
        for word in words:
            if any(x in pretrained for x in [
                word,
                word.lower(),
                re.sub('\d', '0', word.lower())
            ]) and word not in dictionary:
                dictionary[word] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word
'''

'''
 seq_lengths, sort_idx = torch.sort(seq_lengths, descending=True)
        _, unsort_idx = torch.sort(sort_idx)
        seq_embed = seq_embed[sort_idx]

        seq_rep = self.seq_rep(embedded_seqs=seq_embed, seq_lengths=seq_lengths)

        # unsort seq_out
        seq_out = seq_rep[0][unsort_idx]

        bsz, max_seq_len, dim = word_embed.size()
        seq_rep_expand = seq_out.view(bsz, 1, -1).expand(-1, max_seq_len, -1)
        new_embed = torch.cat([seq_rep_expand, word_embed], 2)
'''