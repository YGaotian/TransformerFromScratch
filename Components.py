import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nltk.corpus import brown as brown


class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate: float):
        super().__init__()
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


class StringData:
    def __init__(self):
        self.corpus = brown.words()
        self.vocab = np.append(np.unique([word.lower() for word in self.corpus]), [" ", "<!?UNK?!>"])
        self._word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self._idx2word = {idx: word for word, idx in self._word2idx.items()}

    def getCorpus(self):
        return self.corpus

    def getVocabSize(self):
        return len(self.vocab)

    def word2idx(self, words: list):
        assert len(words) > 0, "No word received."
        return torch.LongTensor([self._word2idx.get(word.lower(), self._word2idx["<!?UNK?!>"]) for word in words])

    def idx2word(self, ids: list):
        assert len(ids) > 0, "No index received."
        return np.array([self._idx2word[idx] for idx in ids])


class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocabulary = nn.Parameter(torch.randn(vocab_size, d_model))

    def forward(self, word_indices: torch.LongTensor):
        return self.vocabulary[word_indices]


class Attention(nn.Module):
    def __init__(self, d_model, head_num, dropout_rate, is_causal=False, max_seq_len=None):
        super().__init__()
        self.model_dim = d_model
        self.head_num = head_num
        self.head_dim = self.model_dim // head_num
        self.dropout_rate = dropout_rate
        self.is_causal = is_causal
        if is_causal:
            assert max_seq_len is not None, "The argument \"max_seq_len\" should be passed in for causal inference."
            self.mask = torch.triu(torch.full([1, 1, max_seq_len, max_seq_len], -torch.inf), diagonal=1)
            self.register_buffer("attnMask", self.mask)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.attn_drop = nn.Dropout(self.dropout_rate)
        self.res_drop = nn.Dropout(self.dropout_rate)

    def forward(self, x_self, x_other=None, cross=False):
        assert (x_other is None) == (not cross), \
            "Cross attention requires these arguments: x_self, x_other, and cross=True."
        if not cross:
            # Shape: (batch_size, sequence_size, model_dim)
            q = self.W_q(x_self)
            k = self.W_k(x_self)
            v = self.W_v(x_self)
        else:
            q = self.W_q(x_self)
            k = self.W_k(x_other)
            v = self.W_v(x_other)
        # Get input shape
        batch_size, seq_len, _ = q.shape
        # Split to n heads
        q_multihead = q.view(batch_size, seq_len, self.head_num, self.head_dim)
        k_multihead = k.view(batch_size, seq_len, self.head_num, self.head_dim)
        v_multihead = v.view(batch_size, seq_len, self.head_num, self.head_dim)
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
        attn_scores = q_multihead @ k_multihead.transpose(-1, -2) / torch.sqrt(self.head_dim)
        # An attention mask for causal tasks.
        if self.is_causal:
            attn_scores += self.mask[:, :, :seq_len, :seq_len]
        # Function "F.softmax" == torch.softmax, != class "nn.Softmax" which needs to be initialized
        attn_weights = F.softmax(attn_scores.float(), dim=-1).type_as(q)  # Use F.softmax by community's convention
        attn_weights = self.attn_drop(attn_weights)
        context_emb = attn_weights @ v_multihead
        # Concatenate multiple heads
        context_emb_T = context_emb.transpose(1, 2)  # Shape: (batch_size, seq_len, head_num, seq_len)
        # Shape: (batch_size, seq_len, model_dim)
        output = context_emb_T.contiguous().view(batch_size, seq_len, self.model_dim)
        output = self.W_o(output)
        output = self.res_drop(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_rate, max_seq_len):
        super().__init__()
        self.position_embedding = nn.Parameter(torch.randn(max_seq_len, d_model))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Shape of x: (batch_size, seq_len, model_dim)
        return self.dropout(x + self.position_embedding[:x.shape[1], :])


class LayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gamma = nn.Parameter(torch.randn(d_model))
        self.beta = nn.Parameter(torch.randn(d_model))

    def forward(self, x):
        normalized = (x - torch.mean(x, dim=-1, keepdim=True)) / (torch.std(x, dim=-1, keepdim=True) + 1e-7)
        return self.gamma * normalized + self.beta


class EncoderLayer(nn.Module):
    def __init__(self, d_model, hidden_dim, head_num, dropout_rate):
        super().__init__()
        self.attention = Attention(d_model, head_num, dropout_rate)
        self.input_normalization = LayerNorm(d_model)
        self.ffn_normalization = LayerNorm(d_model)
        self.ffn = FFN(d_model, hidden_dim, dropout_rate)

    def forward(self, x):
        normalized_input = self.input_normalization(x)
        contextual_embed = self.attention(normalized_input)
        residual_contextual_embed = x + contextual_embed
        normalized_self_context = self.ffn_normalization(residual_contextual_embed)
        ffn_output = self.ffn(normalized_self_context)
        residual_ffn_output = ffn_output + residual_contextual_embed
        return residual_ffn_output


class DecoderLayer(nn.Module):
    def __init__(self, d_model, hidden_dim, head_num, dropout_rate, max_seq_len):
        super().__init__()
        # Self attention needs causal inference, so, pass max_seq_len in it
        self.self_attention = Attention(d_model, head_num, dropout_rate)
        self.input_normalization = LayerNorm(d_model)
        self.cross_normalization = LayerNorm(d_model)
        # Cross attention does not need causal inference
        self.cross_attention = Attention(d_model, head_num, dropout_rate, is_causal=True, max_seq_len=max_seq_len)
        self.ffn_normalization = LayerNorm(d_model)
        self.ffn = FFN(d_model, hidden_dim, dropout_rate)

    def forward(self, x_self, x_other=None, cross=False):
        # Compute self-attention
        normalized_input = self.input_normalization(x_self)
        contextual_embed = self.self_attention(normalized_input)
        residual_self_context = x_self + contextual_embed
        # Compute cross-attention
        normalized_self_context = self.cross_normalization(residual_self_context)
        cross_contextual_embed = self.cross_attention(normalized_self_context, x_other, cross)
        residual_cross_context = residual_self_context + cross_contextual_embed
        # Feed forward network
        normalized_cross_context = self.ffn_normalization(residual_cross_context)
        ffn_output = self.ffn(normalized_cross_context)
        residual_ffn_output = ffn_output + residual_cross_context
        return residual_ffn_output


class Encoder(nn.Module):
    def __init__(self, d_model, hidden_dim, head_num, dropout_rate, layer_num):
        super().__init__()
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, hidden_dim,
                                                          head_num, dropout_rate) for _ in range(layer_num)])
        self.encoder_normalization = LayerNorm(d_model)

    def forward(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        return self.encoder_normalization(x)


class Decoder(nn.Module):
    def __init__(self, d_model, hidden_dim, head_num, dropout_rate, layer_num, max_seq_len):
        super().__init__()
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, hidden_dim, head_num,
                                                          dropout_rate, max_seq_len) for _ in range(layer_num)])
        self.decoder_normalization = LayerNorm(d_model)

    def forward(self, x_self, x_other=None, cross=False):
        for layer in self.decoder_layers:
            x_self = layer(x_self, x_other, cross)
        return self.decoder_normalization(x_self)
