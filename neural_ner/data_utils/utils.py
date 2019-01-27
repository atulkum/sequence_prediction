import codecs
import json
from data_utils.sentence_utils import prepare_sentence
from data_utils.tag_scheme_utils import update_tag_scheme
from data_utils.constant import Constants

def load_sentences(input_format, path, tag_scheme):
    if input_format == 'conll2003':
        return load_sentences_conll(path, tag_scheme)
    else:
        return load_sentences_json(path, tag_scheme)

#for crfsuite formated input
def load_sentences_json(path, tag_scheme):
    sentences = []

    for line in codecs.open(path, 'r', 'utf8'):
        line = line.rstrip()
        if not line:
            continue

        json_data = json.loads(line)
        entities = json_data['entities']
        sentence = [[t] for t in json_data['tokens']]
        curr = 0
        for e in entities:
            name = e['name']
            end = e['end']
            begin = e['begin']

            while curr < begin:
                sentence[curr].append(Constants.ENTITY_OTHER_TAG)
                curr += 1

            sentence[curr].append(Constants.ENTITY_BEGIN + name)
            curr += 1
            while curr <= end:
                sentence[curr].append(Constants.ENTITY_INSIDE + name)
                curr += 1

        while curr < len(sentence):
            sentence[curr].append('O')
            curr += 1

        sentences.append(sentence)

    update_tag_scheme(sentences, tag_scheme)

    return sentences

def load_sentences_conll(path, tag_scheme):
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

def get_chunks(seq):
    col_names = ['name', 'end', 'begin']
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        curr_chunk_type = tok[2:]
        chunk_prefix = tok[:2]
        if tok == Constants.ENTITY_OTHER_TAG:
            if chunk_type is not None:
                chunk = dict(zip(col_names, [chunk_type, chunk_start, i-1]))
                chunks.append(chunk)
                chunk_type, chunk_start = None, None
        elif chunk_prefix == Constants.ENTITY_BEGIN:
            if chunk_type is not None:
                chunk = dict(zip(col_names, [chunk_type, chunk_start, i - 1]))
                chunks.append(chunk)

            chunk_type, chunk_start = curr_chunk_type, i
        elif chunk_prefix == Constants.ENTITY_INSIDE:
            if chunk_type is not None and chunk_type != curr_chunk_type:
                chunk = dict(zip(col_names, [chunk_type, chunk_start, i - 1]))
                chunks.append(chunk)
                chunk_type, chunk_start = None, None

    # end condition
    if chunk_type is not None:
        chunk = dict(zip(col_names, [chunk_type, chunk_start, len(seq) - 1]))
        chunks.append(chunk)
    return chunks



