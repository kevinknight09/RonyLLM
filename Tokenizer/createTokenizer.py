import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="data/dataset.txt",
    model_prefix="tokenizer/tokenizer",
    vocab_size=8000,
    model_type="bpe"
)

print("Tokenizer training completed!")