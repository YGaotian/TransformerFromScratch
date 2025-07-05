from MyTransformer import *

data = StringData()
#
mtx = data.sentence2mtx(["Hello, world!",
                         "My name is Tom.",
                         "How is the weather like today?"])
#
# print(mtx)
#
# id = data.word2idx(["hello", "world", "!", "My", "name", "is", "Tom", "."])
# print(id)
#

# import numpy as np
#
# i = [[0, 1, 0], [1, 0]]
# print(i)

# we = WordEmbedding(data.getVocabSize(), 8)
print(mtx)
# print(we(mtx))
tr = Transformer(32,
                 4,
                 8,
                 data.getVocabSize(),
                 0.3,
                 128,
                 4)

print(tr(mtx, mtx))
