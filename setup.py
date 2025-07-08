from DataGetter import DialogDataset
from MyTransformer import *

D_MODEL = 128
HIDDEN_DIM_MULTIPLE = 4
HEAD_NUM = 8
DROPOUT_RATE = 0.3
MAX_SEQ_LEN = 128
LAYER_NUM = 6

VOCABULARY = Vocabulary()
MODEL = Transformer(D_MODEL, HIDDEN_DIM_MULTIPLE, HEAD_NUM, VOCABULARY.size,
                    DROPOUT_RATE, MAX_SEQ_LEN, LAYER_NUM, VOCABULARY.pad_id)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 500

data_path = "./Data/dialog_data.json"
dialog_dataset = DialogDataset(data_path)


def packer(batch, vocab, device):
    enc_batch, dec_batch = zip(*batch)

    input_enc = vocab.sentence2mtx(list(enc_batch), eos=True).to(device)
    input_dec = vocab.sentence2mtx(list(dec_batch), bos=True).to(device)
    target_lbl = vocab.sentence2mtx(list(dec_batch), eos=True).to(device)

    return input_enc, input_dec, target_lbl


