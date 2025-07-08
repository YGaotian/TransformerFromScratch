from J2APicker import Picker
from torch.utils.data import Dataset


class DialogDataset(Dataset):
    def __init__(self, file_path, tell=False):
        self.data = Picker(file_path, tell)
        self.encoder_data = self.data[:, "A"]
        self.decoder_data = self.data[:, "B"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.encoder_data[item], self.decoder_data[item]



