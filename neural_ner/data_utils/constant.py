class Constants(object):
    _UNK = b"<UNK>"
    _PAD = b"<PAD>"
    _START_VOCAB = [_UNK, _PAD]
    UNK_ID = 0
    PAD_ID = 1
    TAG_PAD_ID = -1

    MAX_CAPS_FEATURE=4

    ENTITY_OTHER_TAG = 'O'
    ENTITY_BEGIN='B-'
    ENTITY_INSIDE = 'I-'
