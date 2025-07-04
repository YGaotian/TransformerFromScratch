from Components import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Transformer(nn.Module):
    def __init__(self, d_model, hidden_dim, head_num, vocab_size, dropout_rate, max_seq_len, layer_num):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.word_emb = WordEmbedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_seq_len, dropout_rate)
        self.encoder = Encoder(d_model, hidden_dim, head_num, dropout_rate, layer_num)
        self.decoder = Decoder(d_model, hidden_dim, head_num, dropout_rate, layer_num, max_seq_len)
        self.linear = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x, target=None):
        word_embedding = self.word_emb(x)
        position_embedding = self.pos_enc(word_embedding)
        encoder_out = self.encoder(position_embedding)
        decoder_out = self.decoder(position_embedding, encoder_out, cross=True)
        if target is not None:
            output = self.linear(decoder_out)
            loss = F.cross_entropy(output.view(-1, output.shape[-1]), target.view(-1))
        else:
            # Get the last word's probability distribution from the distribution matrix of the sequence
            output = self.linear(decoder_out[:, [-1], :])   # Shape: [batch_size, seq_len, d_model]
            loss = None
        return output, loss


