from torch.nn import functional
import numpy as np
import torch
from torch import nn
from torch.nn import init
import math


class Linear(nn.Linear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True):
        super(Linear, self).__init__(in_features, out_features, bias=bias)
        init.orthogonal_(self.weight)


class Linears(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 hiddens,
                 bias=True,
                 activation='tanh'):
        super(Linears, self).__init__()
        assert len(hiddens) > 0

        self.in_features = in_features
        self.out_features = self.output_size = out_features

        in_dims = [in_features] + hiddens[:-1]
        self.linears = nn.ModuleList([Linear(in_dim, out_dim, bias=bias)
                                      for in_dim, out_dim
                                      in zip(in_dims, hiddens)])
        self.output_linear = Linear(hiddens[-1], out_features, bias=bias)
        self.activation = getattr(torch, activation)

    def forward(self, inputs):
        linear_outputs = inputs
        for linear in self.linears:
            linear_outputs = linear.forward(linear_outputs)
            linear_outputs = self.activation(linear_outputs)
        return self.output_linear.forward(linear_outputs)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=.1):
        super(ScaledDotProductAttention, self).__init__()
        self.scale_factor = np.sqrt(d_k)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        # q: [b_size x len_q x d_k]
        # k: [b_size x len_k x d_k]
        # v: [b_size x len_v x d_v] note: (len_k == len_v)
        attn = torch.bmm(q, k.transpose(1, 2)) / self.scale_factor  # attn: [b_size x len_q x len_k]
        if attn_mask is not None:
            # assert attn_mask.size() == attn.size()
            # attn.data.masked_fill_(attn_mask, -10000.0)
            attn += attn_mask

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        outputs = torch.bmm(attn, v)  # outputs: [b_size x len_q x d_v]

        return outputs, attn


class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(d_hid), requires_grad=True)
        self.eps = eps

    def forward(self, z):
        mean = z.mean(dim=-1, keepdim=True, )
        std = z.std(dim=-1, keepdim=True, )
        ln_out = (z - mean.expand_as(z)) / (std.expand_as(z) + self.eps)
        ln_out = self.gamma.expand_as(ln_out) * ln_out + self.beta.expand_as(ln_out)

        return ln_out


class _MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout):
        super(_MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads
        self.w_q = nn.Parameter(torch.FloatTensor(n_heads, d_model, d_k))
        self.w_k = nn.Parameter(torch.FloatTensor(n_heads, d_model, d_k))
        self.w_v = nn.Parameter(torch.FloatTensor(n_heads, d_model, d_v))

        self.attention = ScaledDotProductAttention(d_k, dropout)

        init.xavier_normal_(self.w_q)
        init.xavier_normal_(self.w_k)
        init.xavier_normal_(self.w_v)

    def forward(self, q, k, v, attn_mask=None):
        (d_k, d_v, d_model, n_heads) = (self.d_k, self.d_v, self.d_model, self.n_heads)
        b_size = k.size(0)

        q_s = q.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)  # [n_heads x b_size * len_q x d_model]
        k_s = k.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)  # [n_heads x b_size * len_k x d_model]
        v_s = v.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)  # [n_heads x b_size * len_v x d_model]

        q_s = torch.bmm(q_s, self.w_q).view(b_size * n_heads, -1, d_k)  # [b_size * n_heads x len_q x d_k]
        k_s = torch.bmm(k_s, self.w_k).view(b_size * n_heads, -1, d_k)  # [b_size * n_heads x len_k x d_k]
        v_s = torch.bmm(v_s, self.w_v).view(b_size * n_heads, -1, d_v)  # [b_size * n_heads x len_v x d_v]

        # perform attention, result_size = [b_size * n_heads x len_q x d_v]
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(n_heads, 1, 1)
        outputs, attn = self.attention(q_s, k_s, v_s, attn_mask=attn_mask)

        # return a list of tensors of shape [b_size x len_q x d_v] (length: n_heads)
        return torch.split(outputs, b_size, dim=0), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.attention = _MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.proj = Linear(n_heads * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, q, k, v, attn_mask):
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_v x d_model] note (len_k == len_v)
        residual = q
        # outputs: a list of tensors of shape [b_size x len_q x d_v] (length: n_heads)
        outputs, attn = self.attention(q, k, v, attn_mask=attn_mask)
        # concatenate 'n_heads' multi-head attentions
        outputs = torch.cat(outputs, dim=-1)
        # project back to residual size, result_size = [b_size x len_q x d_model]
        outputs = self.proj(outputs)
        outputs = self.dropout(outputs)

        return self.layer_norm(residual + outputs), attn


class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super(SelfAttention, self).__init__()
        self.attention = MultiHeadAttention(d_model, d_model, d_model, n_heads, dropout)

    def forward(self, q, attn_mask):
        q = q
        k = q
        v = q
        if attn_mask is not None:
            extended_attention_mask = attn_mask.unsqueeze(1)
            extended_attention_mask = extended_attention_mask.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
            attn_mask = (1.0 - extended_attention_mask) * -10000.0
        return self.attention(q, k, v, attn_mask)


class MyMultiheadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout=0.1):
        super(MyMultiheadAttention, self).__init__()
        self.attention = MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)

    def forward(self, q, k, v, k_mask):
        if k_mask is not None:
            extended_attention_mask = k_mask.unsqueeze(1)
            extended_attention_mask = extended_attention_mask.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
            k_mask = (1.0 - extended_attention_mask) * -10000.0
        return self.attention(q, k, v, k_mask)


class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob=0.1):
        super(BertSelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


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


class BertSelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, hidden_dropout_prob=0.1):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(hidden_size, num_attention_heads, hidden_dropout_prob)
        self.output = BertSelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        self_output = self.self(input_tensor, extended_attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

# batch_size = 32
# len_seq = 200
# input_dim = 768
# lable_size = 1
# head = 12
#
# att = MultiHeadAttention(input_dim, input_dim, input_dim, 1, 0.1)
# # selfatt = SelfAttention(input_dim, 1, 0.1)
# inputs = torch.rand(batch_size, len_seq, input_dim)
# q = inputs
# k = inputs
# v = inputs
# masks1 = torch.LongTensor([[[1, 0]]]).transpose(1, 2)
# masks = torch.matmul(masks1, masks1.transpose(1, 2)).byte()
# selfmask = torch.LongTensor([[1, 0]])
# labels = torch.LongTensor([[1, 0]])
# labels = labels.long()
# # o, _ = att(q, k, v, masks)
# # o2, _ = selfatt(inputs, selfmask)
# # print(o2.shape)
#
# # a = torch.rand([1, 3])
# # b = torch.LongTensor([1, 0, 0]).byte()
# # print(a.masked_fill(b, np.inf))
#
# import time
#
# t0 = time.time()
# q = torch.rand(batch_size, len_seq, input_dim)
# k = torch.rand(batch_size, len_seq, input_dim)
# v = k
# k_mask = torch.ones(batch_size, len_seq)
# att = MyMultiheadAttention(input_dim, input_dim, input_dim, head, 0.1)
# o3, _ = att(q, k, v, k_mask)
#
# t1 = time.time()
# print(t1 - t0)
#
#
# t0 = time.time()
# bert_attn = BertAttention(input_dim, head, 0.1)
# a = torch.rand(batch_size, len_seq, input_dim)
# mask = torch.ones(batch_size, len_seq)
# o = bert_attn(a, mask)
# # print(o)
#
# t1 = time.time()
# print(t1 - t0)
