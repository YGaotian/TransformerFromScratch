from Components import *


class Transformer(nn.Module):
    def __init__(self, d_model, hidden_dim_multiple, head_num, vocab_size,
                 dropout_rate, max_seq_len, layer_num, pad_id):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id
        self.word_emb = WordEmbedding(vocab_size, d_model, pad_id)
        self.enc_pos_emb = PositionalEncoding(d_model, max_seq_len, dropout_rate)
        self.dec_pos_emb = PositionalEncoding(d_model, max_seq_len, dropout_rate)
        self.encoder = Encoder(d_model, hidden_dim_multiple, head_num, dropout_rate, layer_num)
        self.decoder = Decoder(d_model, hidden_dim_multiple, head_num, dropout_rate, layer_num, max_seq_len)
        self.linear = nn.Linear(d_model, vocab_size, bias=False)
        # Initialize all parameters
        self.apply(self._init_weights)

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
            output = self.linear(decoder_out[:, [-1], :])   # Shape: [batch_size, 1 (last word of seq), vocab_size]
            loss = None
        return output, loss

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        elif isinstance(module, WordEmbedding):
            nn.init.normal_(module.vocabulary, mean=0.0, std=.02)
            with torch.no_grad():
                module.vocabulary[self.pad_id].fill_(0)

        elif isinstance(module, PositionalEncoding):
            nn.init.normal_(module.position_embedding, mean=0.0, std=.02)


D_MODEL = 128
HIDDEN_DIM_MULTIPLE = 2
HEAD_NUM = 8
DROPOUT_RATE = 0.3
MAX_SEQ_LEN = 128
LAYER_NUM = 4

VOCABULARY = Vocabulary()
MODEL = Transformer(D_MODEL, HIDDEN_DIM_MULTIPLE, HEAD_NUM, VOCABULARY.size,
                    DROPOUT_RATE, MAX_SEQ_LEN, LAYER_NUM, VOCABULARY.pad_id)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
