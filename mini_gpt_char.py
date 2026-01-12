"""

  this code is based on Andrej Karpathy's "Let's build GPT from scratch" Youtube lecture
  (another dataset was used here - see alllines_processed.txt)
  https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7

"""


import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import json


with open('./data/alllines_processed.txt', 'r') as f:
  text = f.read()

print(text[:2000])


torch.manual_seed(5454)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


batch_size = 32
context_length = 256
max_iters = 10000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2


# character tokenizer
chars = sorted(list(set(text)))
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
vocab_size = len(chars)


# encode-decode
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda ints: ''.join([itos[i] for i in ints])


# train-test split (90-10)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train = data[:n+1]
test = data[n+1:]


# load a batch of data
def get_batch(split):
  data = train if split == 'train' else test
  ix = torch.randint(len(data) - context_length, (batch_size,))
  x = torch.stack([data[i:i+context_length] for i in ix])
  y = torch.stack([data[i+1:i+context_length+1] for i in ix])
  x, y = x.to(device), y.to(device)
  return x, y


# calculate loss and perplexity
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        mean_loss = losses.mean().item()
        perplexity = math.exp(mean_loss)
        out[split] = {
            'loss': mean_loss,
            'perplexity': perplexity
        }
    model.train()
    return out


# implementation of nn modules
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

  def __init__(self, head_size):
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

  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = LinearLayer(head_size * num_heads, n_embed)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))
    return out


class FeedForward(nn.Module):
  """ feed-forward network with 2 layers and a non-linear activation in between """

  def __init__(self, n_embed):
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

  def __init__(self, n_embed, n_head):
    super().__init__()
    head_size = n_embed // n_head
    self.mha = MultiHeadAttention(n_head, head_size)
    self.ffn = FeedForward(n_embed)
    self.ln1 = LayerNormalization(n_embed)
    self.ln2 = LayerNormalization(n_embed)

  def forward(self, x):
    x = x + self.mha(self.ln1(x))
    x = x + self.ffn(self.ln2(x))
    return x


class MiniGPT(nn.Module):
  """ entire decoder applying the transformer block times n_layer"""

  def __init__(self):
    super().__init__()
    self.token_embed_table = nn.Embedding(vocab_size, n_embed)
    self.position_embed_table = nn.Embedding(context_length, n_embed)
    self.blocks = nn.Sequential(*[Block(n_embed, n_head) for _ in range(n_layer)])
    self.lin = LinearLayer(n_embed, vocab_size)

  def forward(self, idx, targets=None):
    b, t = idx.shape
    tok_emb = self.token_embed_table(idx) # (b, t, e)
    pos_emb = self.position_embed_table(torch.arange(t, device=device)) # (t, e)
    x = tok_emb + pos_emb # (b, t, e)
    x = self.blocks(x) # (b, t, e)
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
        idx_cond = idx[:, -context_length:]
        logits, loss = self(idx_cond)
        logits = logits[:, -1, :] / temperature # (b, vs)
        probs = F.softmax(logits, dim=-1) # (b, vs)
        idx_next = torch.multinomial(probs, num_samples=1) # (b, 1)
        idx = torch.cat((idx, idx_next), dim=1) # (b, t+1)
    return idx


model = MiniGPT()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


results = []
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        results.append(losses)

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
hamlet_sample1 = "HAMLET: Not so, my lord, I am too much i' the sun.\n\nROMEO: "
hamlet_sample2 = "HAMLET: To be, or not to be, that is the question.\n\nROMEO: "
context = torch.tensor(encode(hamlet_sample2), dtype=torch.long).unsqueeze(0).to(device)
open('char_large_sample.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))


# save loss dict
with open('char_large_results.json', 'w') as f:
    json.dump(results, f, indent=4)