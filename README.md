# RonyLLM

RonyLLM is a custom, lightweight Large Language Model (LLM) built natively from scratch using Python and PyTorch. It features a custom `MiniGPT` architecture based on the Transformer Encoder design and provides an end-to-end pipeline covering data preparation, model training, and text generation through a web interface.

## Project Structure

The repository is organized as follows:

- **`Model/`**: Contains the model architecture implementation and the saved model weights.
  - `ronyGpt.py`: Defines the `MiniGPT` class, an architecture using `nn.Embedding`, `nn.TransformerEncoderLayer`, and a linear projection head.
  - `ronyGpt.pt`: The trained model weights.
- **`Tokenizer/`**: Contains the tokenization models.
  - `tokenizer.model`: A custom SentencePiece model used for encoding and decoding text.
- **`Data/`**: Directory meant to hold the training data.
  - `dataset.txt`: A text file containing the raw training data.
- **`dataSetPrepare.py`**: A script that reads `data/dataset.txt` and encodes it into tensors using the SentencePiece tokenizer.
- **`train.py`**: The training script. It loads the text data, initializes the `MiniGPT` model, and trains it using an Adam optimizer and Cross Entropy Loss. The trained model is saved to `model/ronyGpt.pt`. It has resume capabilities and saves checkpoints on keyboard interrupt.
- **`generate.py`**: The inference script. It loads the tokenizer and the trained model, then launches a user-friendly Gradio web interface to allow interactive text generation based on user prompts.

## Prerequisites

Make sure you have the following dependencies installed:

```bash
pip install torch sentencepiece gradio
```

## Usage

### 1. Data Preparation

Place your raw text training data inside the `Data` directory and name it `dataset.txt`. Make sure the SentencePiece tokenizer model is available at `tokenizer/tokenizer.model`.

You can test the encoding process using:

```bash
python dataSetPrepare.py
```

### 2. Training the Model

To train the LLM, simply run:

```bash
python train.py
```

The script will:
- Load your dataset and encode it.
- Initialize the `MiniGPT` model (or load existing weights).
- Train using batches of sequences (default batch size: 32, sequence length: 64).
- Save the trained weights to `Model/ronyGpt.pt` automatically upon interruption (`Ctrl+C`).

### 3. Text Generation

Once the model is trained, you can interact with it using a web UI:

```bash
python generate.py
```

This will launch a Gradio interface in your web browser. You can enter a prompt and the model will auto-regressively generate text continuation using temperature sampling.

## Architecture details

The internal `MiniGPT` model uses a parameterizable sequence model built from PyTorch's `nn.TransformerEncoder`. 

- **Vocabulary Size:** 8000 (configurable via SentencePiece)
- **Embedding Dimension:** 256
- **Attention Heads:** 4
- **Transformer Layers:** 4

During generation, the model leverages multinomial sampling on softmax probabilities scaled by a temperature factor to introduce diversity in the generated text.
