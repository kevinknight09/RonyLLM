import torch
import torch.nn as nn
import sentencepiece as spm

from Model import ronyGpt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

sp = spm.SentencePieceProcessor()
sp.load("tokenizer/tokenizer.model")

with open("data/dataset.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokens = sp.encode(text)
data = torch.tensor(tokens)

vocab_size = 8000
model = ronyGpt.MiniGPT(vocab_size)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()

batch_size = 32
seq_len = 64

model.load_state_dict(torch.load("model/ronyGpt.pt"))

try:

    for step in range(10000):

        idx = torch.randint(len(data)-seq_len, (batch_size,))

        x = torch.stack([data[i:i+seq_len] for i in idx]).to(device)
        y = torch.stack([data[i+1:i+seq_len+1] for i in idx]).to(device)

        logits = model(x)

        loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(step, loss.item())

except KeyboardInterrupt:

    print("Training interrupted. Saving model...")
    torch.save(model.state_dict(), "model/ronyGpt.pt")