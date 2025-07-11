from MyTransformer import *

D_MODEL = 256
HIDDEN_DIM_MULTIPLE = 2
HEAD_NUM = 4
DROPOUT_RATE = 0.1
MAX_SEQ_LEN = 256
LAYER_NUM = 4

VOCABULARY = Vocabulary()
MODEL = Transformer(D_MODEL, HIDDEN_DIM_MULTIPLE, HEAD_NUM, VOCABULARY.size,
                    DROPOUT_RATE, MAX_SEQ_LEN, LAYER_NUM, VOCABULARY.pad_id)

cuda_ready = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda_ready else "cpu")
