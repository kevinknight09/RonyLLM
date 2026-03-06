import torch
import torch.nn as nn
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("tokenizer/tokenizer.model")

with open("data/dataset.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokens = sp.encode(text)
data = torch.tensor(tokens)