from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

text = "\n".join(dataset["train"]["text"])

with open("Data/dataset.txt", "w", encoding="utf-8") as f:
    f.write(text)