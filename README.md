# Emoji Translation Dataset Creator

This project is inspired by the paper **[Emojinize](https://arxiv.org/abs/2403.03857)** and generates a small supervised fine-tuning (SFT) dataset for training models to translate words inside `< >` into emojis based on context.

The pipeline:
- Loads input sentences from `data/input_examples.txt`
- Queries an LLM (via OpenRouter) using few-shot examples
- Produces JSON emoji translations
- Saves a HuggingFace dataset on disk
- Creates a clean CSV preview with only:  
  **input_text, emoji**

---

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt

2. **Add your API key**
   Similar to .env.example, create a file in the root directory called "env" with the line:
   OPENROUTER_API_KEY=your-key-here

2. **Model used**
   google/gemma-3-27b-it:free (can be changed by changing the "MODEL_NAME" parameter under config.py)


## Usage

1. **Add your inputs**
One sentence per line, add with the targeted word surrounded by "< >" to the file:
data/input_examples.txt

2. **Run the dataset builder**
python build_dataset.py

3. **Outputs**
HuggingFace dataset (compatible with Hugging Face SFTTrainer from trl) → data/emoji_sft_dataset/
CSV preview → data/emoji_sft_dataset_preview.csv
Printed preview in terminal


## Project Structure

build_dataset.py        # Main script to generate dataset
config.py               # System prompt, few-shot examples, paths, model config
utils.py                # LLM queries + dataset helpers
data/
  input_examples.txt
  emoji_sft_dataset/
  emoji_sft_dataset_preview.csv
.env.example
requirements.txt



