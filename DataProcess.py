from torch.utils.data import Dataset
import json


class DialogDataset(Dataset):
    def __init__(self, file_path, tell=False):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                self.data = json.load(file)
            if tell and len(self.data):
                print(f"{len(self.data)} pairs of data loaded.")
        except Exception as e:
            raise e

        self.encoder_data = [dialog[0] for dialog in self.data]
        self.decoder_data = [dialog[1] for dialog in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.encoder_data[item], self.decoder_data[item]


def packer(batch, vocab, device):
    enc_batch, dec_batch = zip(*batch)

    input_enc = vocab.sentence2mtx(list(enc_batch), eos=True).to(device)
    input_dec = vocab.sentence2mtx(list(dec_batch), bos=True).to(device)
    target_lbl = vocab.sentence2mtx(list(dec_batch), eos=True).to(device)

    return input_enc, input_dec, target_lbl

