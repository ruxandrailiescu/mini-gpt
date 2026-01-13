import torch
import torch.nn as nn
from torch.nn import functional as F
import math


torch.manual_seed(5454)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LinearLayer(nn.Module):
  """ affine linear transformation """

  def __init__(self, in_features, out_features, bias=True):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.weight = nn.Parameter(torch.randn(out_features, in_features))

    if bias:
      self.bias = nn.Parameter(torch.zeros(out_features))
    else:
      self.register_parameter('bias', None)

    self.reset_parameters()

  def reset_parameters(self):
    sqrt_k = 1. / math.sqrt(self.in_features)
    self.weight.data.uniform_(-sqrt_k, sqrt_k)
    if self.bias is not None:
      self.bias.data.uniform_(-sqrt_k, sqrt_k)

  def forward(self, x):
    x = x @ self.weight.T
    if self.bias is not None:
      x = x + self.bias
    return x


class Head(nn.Module):
  """ single head of self-attention """

  def __init__(self, head_size, n_embed, context_length, dropout):
    super().__init__()
    self.key = LinearLayer(n_embed, head_size, bias=False)  # using the custom linear layer
    self.query = LinearLayer(n_embed, head_size, bias=False)
    self.value = LinearLayer(n_embed, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))  # attention mask
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    # input: (batch, time-step, embed)
    # output: (batch, time-step, head_size)
    b,t,e = x.shape
    k = self.key(x) # (b, t, hs)
    q = self.query(x) # (b, t, hs)

    # scaled dot-attention
    w = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (b, t, hs) @ (b, t, hs) --> (b, t, t)
    w = w.masked_fill(self.tril[:t, :t] == 0, float('-inf'))
    w = F.softmax(w, dim=-1)
    w = self.dropout(w)

    v = self.value(x) # (b, t, hs)
    out = w @ v # (b, t, t) @ (b, t, hs) --> (b, t, hs)
    return out


class MultiHeadAttention(nn.Module):
  """ multiple self-attention heads in parallel """

  def __init__(self, num_heads, head_size, n_embed, context_length, dropout):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size, n_embed, context_length, dropout) for _ in range(num_heads)])
    self.proj = LinearLayer(head_size * num_heads, n_embed)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))
    return out


class FeedForward(nn.Module):
  """ feed-forward network with 2 layers and a non-linear activation in between """

  def __init__(self, n_embed, dropout):
    super().__init__()
    self.net = nn.Sequential(
        LinearLayer(n_embed, n_embed*4),
        nn.ReLU(),
        LinearLayer(n_embed*4, n_embed),
        nn.Dropout(dropout),
    )

  def forward(self, x):
    return self.net(x)


class LayerNormalization(nn.Module):
  """ computes statistics across a single sample (features dimension) """

  def __init__(self, dim, eps=1e-5):
    super().__init__()
    self.eps = eps
    self.gamma = nn.Parameter(torch.ones(dim))
    self.beta = nn.Parameter(torch.zeros(dim))

  def forward(self, x):
    xmean = x.mean(-1, keepdim=True)
    xvar = x.var(-1, keepdim=True)
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
    self.out = self.gamma * xhat + self.beta
    return self.out


class Block(nn.Module):
  """ transformer block """

  def __init__(self, n_embed, n_head, dropout, context_length):
    super().__init__()
    head_size = n_embed // n_head
    self.mha = MultiHeadAttention(n_head, head_size, n_embed, context_length, dropout)
    self.ffn = FeedForward(n_embed, dropout)
    self.ln1 = LayerNormalization(n_embed)
    self.ln2 = LayerNormalization(n_embed)

  def forward(self, x):
    x = x + self.mha(self.ln1(x))
    x = x + self.ffn(self.ln2(x))
    return x


class MiniGPT(nn.Module):
  """ entire decoder applying the transformer block times n_layer"""

  def __init__(self, n_layer, n_head, n_embed, context_length, vocab_size, dropout):
    super().__init__()
    self.context_length = context_length
    self.token_embed_table = nn.Embedding(vocab_size, n_embed)
    self.position_embed_table = nn.Embedding(context_length, n_embed)
    self.blocks = nn.Sequential(*[Block(n_embed, n_head, dropout, context_length) for _ in range(n_layer)])
    self.layern = LayerNormalization(n_embed)
    self.lin = LinearLayer(n_embed, vocab_size)

  def forward(self, idx, targets=None):
    b, t = idx.shape
    tok_emb = self.token_embed_table(idx) # (b, t, e)
    pos_emb = self.position_embed_table(torch.arange(t, device=device)) # (t, e)
    x = tok_emb + pos_emb # (b, t, e)
    x = self.blocks(x) # (b, t, e)
    x = self.layern(x)
    logits = self.lin(x) # (b, t, vocab_size)

    if targets is None:
        loss = None
    else:
        b, t, vs = logits.shape
        logits = logits.view(b*t, vs)
        targets = targets.view(b*t)
        loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens, temperature=1.0):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -self.context_length:]
        logits, loss = self(idx_cond)
        logits = logits[:, -1, :] / temperature # (b, vs)
        probs = F.softmax(logits, dim=-1) # (b, vs)
        idx_next = torch.multinomial(probs, num_samples=1) # (b, 1)
        idx = torch.cat((idx, idx_next), dim=1) # (b, t+1)
    return idx