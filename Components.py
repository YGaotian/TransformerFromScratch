import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nltk.corpus import brown as brown
from DebugMode import log


class FFN(nn.Module):
    def __init__(self, dim, hidden_dim_multiple, dropout_rate: float):
        super().__init__()
        hidden_dim = dim * hidden_dim_multiple
        self.input_layer = nn.Linear(dim, hidden_dim, bias=False)
        self.out_layer = nn.Linear(hidden_dim, dim, bias=False)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        layer_1 = self.input_layer(x)
        layer_2 = self.activation(layer_1)
        layer_3 = self.out_layer(layer_2)
        output = self.dropout(layer_3)
        return output


class Vocabulary:
    def __init__(self):
        self._corpus = brown.words()
        self._vocab = np.append(np.unique([word.lower() for word in self._corpus]),
                                ["<bos>", "<eos>", "<unk>", "<pad>"]).tolist()
        self._word2idx = {word: idx for idx, word in enumerate(self._vocab)}
        self._idx2word = {idx: word for word, idx in self._word2idx.items()}
        self._pad_id = self._word2idx["<pad>"]
        self._eos_id = self._word2idx["<eos>"]
        self._size = len(self._vocab)

    @property
    def pad_id(self):
        return self._pad_id

    @property
    def eos_id(self):
        return self._eos_id

    @property
    def corpus(self):
        return self._corpus

    @property
    def size(self):
        return self._size

    def word2idx(self, words: list):
        assert len(words) > 0, "No word received."
        return [self._word2idx.get(word.lower(), self._word2idx["<unk>"]) for word in words]

    def sentence2mtx(self, sentences: list[str], bos=False, eos=False, to_torch=True):
        word_seq_arr = []
        max_pad_len = 0
        padded_sentences = []
        for s in sentences:
            word_list = nltk.word_tokenize(s)
            if bos:
                word_list = ["<bos>"] + word_list
            if eos:
                word_list = word_list + ["<eos>"]
            log(word_list)
            if max_pad_len < len(word_list):
                max_pad_len = len(word_list)
            word_seq_arr.append(self.word2idx(word_list))
        for seq in word_seq_arr:
            pad_num = max_pad_len - len(seq)
            padded_seq = seq + pad_num * [self.pad_id]
            padded_sentences.append(padded_seq)
        if to_torch:
            padded_sentences = torch.tensor(padded_sentences)
        return padded_sentences

    def idx2word(self, ids: list):
        assert len(ids) > 0, "No index received."
        return [self._idx2word[idx] for idx in ids]


class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, pad_id):
        super().__init__()
        self.pad_id = pad_id
        padding_emb = torch.zeros(1, d_model)
        self.vocabulary = nn.Parameter(
            torch.cat([torch.randn(vocab_size - 1, d_model), padding_emb], dim=0)
        )

    def forward(self, word_indices):
        padding_mask = (word_indices == self.pad_id)
        return self.vocabulary[word_indices], padding_mask


class Attention(nn.Module):
    def __init__(self, d_model, head_num, dropout_rate, is_causal=False, max_seq_len=None):
        super().__init__()
        assert d_model % head_num == 0, "d_model must be an integer multiple of head_num."
        self.model_dim = d_model
        self.head_num = head_num
        self.head_dim = self.model_dim // head_num
        self.dropout_rate = dropout_rate
        self.is_causal = is_causal
        if is_causal:
            assert max_seq_len is not None, \
                "The argument \"max_seq_len\" should be passed in for causal inference."
            self.causal_mask = torch.triu(torch.full([1, 1, max_seq_len, max_seq_len], -torch.inf), diagonal=1)
            self.register_buffer("causal_attn_mask", self.causal_mask)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.attn_drop = nn.Dropout(self.dropout_rate)
        self.res_drop = nn.Dropout(self.dropout_rate)

    def forward(self, x_self, x_other, padding_mask):
        is_cross = not (x_self is x_other)
        # Shape: (batch_size, sequence_size, model_dim)
        q = self.W_q(x_self)
        k = self.W_k(x_other if is_cross else x_self)
        v = self.W_v(x_other if is_cross else x_self)
        # Get input shape
        q_batch_size, q_seq_len, _ = q.shape
        kv_batch_size, kv_seq_len, _ = k.shape
        assert q_batch_size == kv_batch_size, \
            "Encoder input and decoder input must have the same batch size."
        batch_size = q_batch_size
        log("q Shape: ", q.shape)
        log("k Shape: ", k.shape)
        # Split to n heads
        q_multihead = q.view(batch_size, q_seq_len, self.head_num, self.head_dim)
        k_multihead = k.view(batch_size, kv_seq_len, self.head_num, self.head_dim)
        v_multihead = v.view(batch_size, kv_seq_len, self.head_num, self.head_dim)
        # Matmul operates on the last 2 dimensions of a Tensor, and the remaining dimensions are considered batches,
        # which will be merged element-wise. For the current case, we need (seq_len, head_dim) @ (head_dim, seq_len),
        # but our tensors' last 2 dimensions are head_num and head_dim. So, we need to switch dim_1 and dim_2.
        # Shape: (batch_size, head_num, seq_len, head_dim)
        q_multihead = q_multihead.transpose(1, 2)
        k_multihead = k_multihead.transpose(1, 2)
        v_multihead = v_multihead.transpose(1, 2)
        # For each head, every word in a sequence would have a dot product for all words contained in that sequence.
        # Therefore, there are (seq_len * seq_len) dot products in every head.
        # Shape: (batch_size, head_num, seq_len, seq_len)
        attn_scores = q_multihead @ k_multihead.transpose(-1, -2) / torch.sqrt(torch.tensor(self.head_dim))
        # Attention mask for causal tasks.
        if self.is_causal and not is_cross:
            attn_scores += self.causal_attn_mask[:, :, :q_seq_len, :kv_seq_len]
            log(attn_scores.shape)
        col_extended_padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
        log(col_extended_padding_mask.shape)
        attn_scores += (torch.zeros_like(col_extended_padding_mask, dtype=torch.float)
                             .masked_fill_(col_extended_padding_mask, -torch.inf))
        # Function "F.softmax" == torch.softmax, != class "nn.Softmax" which needs to be initialized
        attn_weights = F.softmax(attn_scores.float(), dim=-1).type_as(q)  # Use F.softmax by community's convention
        attn_weights = self.attn_drop(attn_weights)
        context_emb = attn_weights @ v_multihead
        # Concatenate multiple heads
        context_emb_T = context_emb.transpose(1, 2)  # Shape: (batch_size, seq_len, head_num, seq_len)
        # Shape: (batch_size, seq_len, model_dim)
        output = context_emb_T.contiguous().view(batch_size, q_seq_len, self.model_dim)
        output = self.W_o(output)
        output = self.res_drop(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len, dropout_rate):
        super().__init__()
        self.position_embedding = nn.Parameter(torch.randn([max_seq_len, d_model]))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Shape of x: (batch_size, seq_len, model_dim)
        return self.dropout(x + self.position_embedding[:x.shape[1], :])


class LayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model, dtype=torch.float))
        self.beta = nn.Parameter(torch.zeros(d_model, dtype=torch.float))

    def forward(self, x):
        normalized = (x - torch.mean(x, dim=-1, keepdim=True)) / (torch.std(x, dim=-1, keepdim=True) + 1e-7)
        return self.gamma * normalized + self.beta


class EncoderLayer(nn.Module):
    def __init__(self, d_model, hidden_dim_multiple, head_num, dropout_rate):
        super().__init__()
        self.attention = Attention(d_model, head_num, dropout_rate)
        self.input_normalization = LayerNorm(d_model)
        self.ffn_normalization = LayerNorm(d_model)
        self.ffn = FFN(d_model, hidden_dim_multiple, dropout_rate)

    def forward(self, x, self_padding_mask):
        normalized_input = self.input_normalization(x)
        contextual_embed = self.attention(normalized_input, normalized_input, self_padding_mask)
        residual_contextual_embed = x + contextual_embed
        normalized_self_context = self.ffn_normalization(residual_contextual_embed)
        ffn_output = self.ffn(normalized_self_context)
        residual_ffn_output = ffn_output + residual_contextual_embed
        return residual_ffn_output


class DecoderLayer(nn.Module):
    def __init__(self, d_model, hidden_dim_multiple, head_num, dropout_rate, max_seq_len):
        super().__init__()
        # Self attention needs causal inference, so, pass max_seq_len in it
        self.self_attention = Attention(d_model, head_num, dropout_rate, is_causal=True, max_seq_len=max_seq_len)
        self.input_normalization = LayerNorm(d_model)
        self.cross_normalization = LayerNorm(d_model)
        # Cross attention does not need causal inference
        self.cross_attention = Attention(d_model, head_num, dropout_rate)
        self.ffn_normalization = LayerNorm(d_model)
        self.ffn = FFN(d_model, hidden_dim_multiple, dropout_rate)

    def forward(self, x_self, self_padding_mask, x_other, other_padding_mask):
        # Compute self-attention
        normalized_input = self.input_normalization(x_self)
        log("computing self attn")
        contextual_embed = self.self_attention(normalized_input, normalized_input, self_padding_mask)
        log("self attn done")
        residual_self_context = x_self + contextual_embed
        # Compute cross-attention
        normalized_self_context = self.cross_normalization(residual_self_context)
        log("computing cross attn")
        cross_contextual_embed = self.cross_attention(normalized_self_context,
                                                      x_other,
                                                      other_padding_mask)
        log("cross attn done")
        residual_cross_context = residual_self_context + cross_contextual_embed
        # Feed forward network
        normalized_cross_context = self.ffn_normalization(residual_cross_context)
        ffn_output = self.ffn(normalized_cross_context)
        residual_ffn_output = ffn_output + residual_cross_context
        return residual_ffn_output


class Encoder(nn.Module):
    def __init__(self, d_model, hidden_dim_multiple, head_num, dropout_rate, layer_num):
        super().__init__()
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, hidden_dim_multiple,
                                                          head_num, dropout_rate) for _ in range(layer_num)])
        self.encoder_normalization = LayerNorm(d_model)

    def forward(self, x, self_padding_mask):
        for layer in self.encoder_layers:
            x = layer(x, self_padding_mask)
        return self.encoder_normalization(x)


class Decoder(nn.Module):
    def __init__(self, d_model, hidden_dim_multiple, head_num, dropout_rate, layer_num, max_seq_len):
        super().__init__()
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, hidden_dim_multiple, head_num,
                                                          dropout_rate, max_seq_len) for _ in range(layer_num)])
        self.decoder_normalization = LayerNorm(d_model)

    def forward(self, x_self, self_padding_mask, x_other, other_padding_mask):
        for layer in self.decoder_layers:
            x_self = layer(x_self, self_padding_mask, x_other, other_padding_mask)
        return self.decoder_normalization(x_self)
