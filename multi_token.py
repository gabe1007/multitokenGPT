import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants from main.py
BATCH_SIZE = 128
BLOCK_SIZE = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_EMBED = 192
N_HEADS = 3
DROPOUT = 0.2
N_LAYER = 4
LEARNING_RATE = 3e-3
MAX_ITERS = 10000
EVAL_INTERVAL = 500
EVAL_ITERS = 200

# New constant for multi-token prediction
MULTI_TOKEN_SIZE = 4  # Number of tokens to predict at once

class MultiTokenHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiTokenMultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([MultiTokenHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, N_EMBED)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class MultiTokenFeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)

class MultiTokenBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiTokenMultiHeadAttention(n_head, head_size)
        self.ffwd = MultiTokenFeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MultiTokenGPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBED)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBED)
        self.blocks = nn.Sequential(*[MultiTokenBlock(N_EMBED, n_head=N_HEADS) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBED)
        self.lm_head = nn.Linear(N_EMBED, vocab_size * MULTI_TOKEN_SIZE)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B, T, MULTI_TOKEN_SIZE, -1)
            targets = targets.view(B, T, MULTI_TOKEN_SIZE)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens // MULTI_TOKEN_SIZE):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :].view(-1, MULTI_TOKEN_SIZE, VOCAB_SIZE)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs.view(-1, VOCAB_SIZE), num_samples=1).view(-1, MULTI_TOKEN_SIZE)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def get_batch(split, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE):
    data = train if split == 'train' else val
    ix = torch.randint(len(data) - block_size - MULTI_TOKEN_SIZE, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size+MULTI_TOKEN_SIZE] for i in ix])
    y = y.view(batch_size, block_size, MULTI_TOKEN_SIZE)
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

if __name__ == '__main__':
    # Load and preprocess data
    with open('./40k.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    VOCAB_SIZE = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train = data[:n]
    val = data[n:]

    # Initialize model and optimizer
    model = MultiTokenGPTLanguageModel(VOCAB_SIZE).to(DEVICE)
    print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for iter in range(MAX_ITERS):
        if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Generate sample text
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

    # Save the model
    torch.save(model.state_dict(), "multi_token_model.pth")