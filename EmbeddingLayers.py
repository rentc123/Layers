from torch import nn
import torch
from layers.AttentionLayers import MyMultiheadAttention


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class WordEmbeddingLayer(nn.Module):
    """Construct the embeddings from word, position embeddings.
    """

    def __init__(self, vocab_size, hidden_size, max_len, dropout_prob=0.1):
        super(WordEmbeddingLayer, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_len, hidden_size)
        self.token_type_embeddings = nn.Embedding(2, hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_ids, token_type_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ChWordEmbeddingLayer(nn.Module):
    """Construct the embeddings from char and word, position embeddings.
    then output shape like as char tensor shape
    """

    def __init__(self, ch_vocab_size, word_vocab_size, hidden_size, ch_max_len, n_heads=1, dropout_prob=0.1):
        super(ChWordEmbeddingLayer, self).__init__()
        self.ch_embeddings = nn.Embedding(ch_vocab_size, hidden_size, padding_idx=0)
        self.word_embeddings = nn.Embedding(word_vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(ch_max_len, hidden_size)
        self.token_type_embeddings = nn.Embedding(2, hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)
        self.multi_attn = MyMultiheadAttention(hidden_size, hidden_size, hidden_size, n_heads)

    def forward(self, ch_input_ids, word_input_ids, word_mask=None, token_type_ids=None):
        seq_length = ch_input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=ch_input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(ch_input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(ch_input_ids)

        chs_embeddings = self.ch_embeddings(ch_input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = chs_embeddings + position_embeddings + token_type_embeddings
        words_embeddings = self.word_embeddings(word_input_ids)

        embeddings, _ = self.multi_attn(embeddings, words_embeddings, words_embeddings, word_mask)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


if __name__ == '__main__':
    len_ch = 128
    len_word = 100
    dim = 200
    a = ChWordEmbeddingLayer(80, 80, dim, len_ch, n_heads=4, dropout_prob=0.1)
    ch_input_ids = torch.ones(2, len_ch).long()
    word_input_ids = torch.ones(2, len_word).long()
    word_mask = torch.ones(2, len_word).long()
    e = a(ch_input_ids, word_input_ids, word_mask)
    print(e.shape)
