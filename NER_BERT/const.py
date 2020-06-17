from torch.nn import CrossEntropyLoss

pad_token_id = 0

ENTITY_OTHER = 'O'
ENTITY_BEGIN = 'B-'
ENTITY_CONT = 'I-'
ENTITY_SINGLE = 'S-'
ENTITY_END = 'E-'

UNK_INTENT = 'unknown'

label_pad_id = CrossEntropyLoss().ignore_index
sep_token = "[SEP]"
cls_token = "[CLS]"
cls_token_segment_id = 0
sequence_a_segment_id = 0
pad_token_segment_id = 0

MODEL_TYPE = 'bert-base-uncased'

ENTITY_NAMES=[
"Appeal_to_Authority",
"Appeal_to_fear-prejudice",
"Black-and-White_Fallacy",
"Causal_Oversimplification",
"Doubt",
"Exaggeration,Minimisation",
"Flag-Waving",
"Loaded_Language",
"Name_Calling,Labeling",
"Obfuscation,Intentional_Vagueness,Confusion",
"Repetition",
"Slogans",
"Thought-terminating_Cliches",
"Whataboutism,Straw_Men,Red_Herring",
"Bandwagon,Reductio_ad_hitlerum"
]

_UNK = "<UNK>"
_PAD = "<PAD>"
_START_VOCAB = [_UNK, _PAD]
UNK_ID = 0
PAD_ID = 1