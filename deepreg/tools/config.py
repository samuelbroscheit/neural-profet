PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
BOS_TOKEN = '<s>'
EOS_TOKEN = '<\s>'
BOP_TOKEN = '<p>'
EOP_TOKEN = '<\p>'
BOD_TOKEN = '<d>'
EOD_TOKEN = '<\d>'

PAD, UNK, BOS, EOS, BOP, EOP, BOD, EOD = [0, 1, 2, 3, 4, 5, 6, 7]

SPECIAL_TOKEN_IDS = [PAD, UNK, BOS, EOS, BOP, EOP, BOD, EOD]
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, BOP_TOKEN, EOP_TOKEN, BOD_TOKEN, EOD_TOKEN]

for id,tok in zip(SPECIAL_TOKEN_IDS, SPECIAL_TOKENS):
    assert SPECIAL_TOKENS[id] == tok

LANGUAGE_TOKENS = lambda lang: '<%s>' % lang
