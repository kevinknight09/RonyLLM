import torch
import sentencepiece as spm
import gradio as gr

from Model import ronyGpt

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer
sp = spm.SentencePieceProcessor()
sp.load("tokenizer/tokenizer.model")

vocab_size = sp.get_piece_size()

# Load model
model = ronyGpt.MiniGPT(vocab_size).to(device)
model.load_state_dict(torch.load("model/ronyGpt.pt", map_location=device))
model.eval()

def generate_text(prompt):
    tokens = sp.encode(prompt)
    tokens = torch.tensor(tokens).unsqueeze(0).to(device)

    for _ in range(50):
        tokens = tokens[:, -128:]
        with torch.no_grad():
            logits = model(tokens)
            temperature = 0.8
            probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)

    return sp.decode(tokens[0].tolist())

interface = gr.Interface(
    fn=generate_text,
    inputs="text",
    outputs="text",
    title=" RonyGPT native LLM made using Python",
    description="Enter a prompt and generate text."
)

interface.launch()