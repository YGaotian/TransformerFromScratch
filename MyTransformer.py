from Components import *


class Transformer(nn.Module):
    def __init__(self, d_model, hidden_dim, head_num, vocab_size,
                 dropout_rate, max_seq_len, layer_num, pad_id=-1):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id
        self.word_emb = WordEmbedding(vocab_size, d_model, pad_id)
        self.enc_pos_emb = PositionalEncoding(d_model, max_seq_len, dropout_rate)
        self.dec_pos_emb = PositionalEncoding(d_model, max_seq_len, dropout_rate)
        self.encoder = Encoder(d_model, hidden_dim, head_num, dropout_rate, layer_num)
        self.decoder = Decoder(d_model, hidden_dim, head_num, dropout_rate, layer_num, max_seq_len)
        self.linear = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x_enc, x_dec, target=None):
        enc_emb, enc_padding_mask = self.word_emb(x_enc)
        dec_emb, dec_padding_mask = self.word_emb(x_dec)
        positional_enc_emb = self.enc_pos_emb(enc_emb)
        positional_dec_emb = self.dec_pos_emb(dec_emb)
        log("computing encoder")
        encoder_out = self.encoder(positional_enc_emb, enc_padding_mask)
        log("computing decoder")
        decoder_out = self.decoder(positional_dec_emb, dec_padding_mask, encoder_out, enc_padding_mask)
        if target is not None:
            output = self.linear(decoder_out)
            loss = F.cross_entropy(output.view(-1, output.shape[-1]), target.view(-1), ignore_index=self.pad_id)
        else:
            # Get the last word's probability distribution from the distribution matrix of the sequence
            output = self.linear(decoder_out[:, [-1], :])   # Shape: [batch_size, seq_len, d_model]
            loss = None
        return output, loss


