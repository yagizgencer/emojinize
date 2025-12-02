# Emoji Translation Dataset Creator

This project is inspired by the paper **[Emojinize](https://arxiv.org/abs/2403.03857)** and generates a small supervised fine-tuning (SFT) dataset for training models to translate words inside `< >` into emojis based on context.

The pipeline:
- Loads input sentences from `data/input_examples.txt`
- Queries an LLM (via `OpenRouter`) using few-shot examples
- Produces JSON emoji translations
- Saves a `Hugging Face` dataset on disk
- Creates a clean CSV preview with only `input_text` and `emoji` fields

---

## Setup

1. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt

2. **Add your API key**  
   Similar to `.env.example`, create a file in the root directory called `.env` with the line:  
   `OPENROUTER_API_KEY=your-key-here`

2. **Model Used**  
   `google/gemma-3-27b-it:free` (can be changed by modifying the `MODEL_NAME` parameter under `config.py`)
   
   You can browse available OpenRouter models [here](https://openrouter.ai/models).


## Usage

1. **Add Your Inputs**  
Add one example per line with the targeted word surrounded by `< >` to the file `data/input_examples.txt`.

2. **Run the Dataset Builder**  
   ```bash
   python build_dataset.py

4. **Outputs**  
- **HuggingFace Dataset (compatible with TRL’s SFTTrainer):**  
  `data/emoji_sft_dataset/`
- **CSV Preview:**  
  `data/emoji_sft_dataset_preview.csv`
- **A preview of the dataset** is also printed in the terminal

----------
# Brainstorming About Different Transformer Architectures
Here, we will discuss possible architectures and their pros and cons in the context of using our dataset to fine-tune/align them for the text to emoji translation task. For each of the architectures, we will first introduce what they do, then briefly explain how our dataset could be used to fine-tune them for our task and finally list some pros and cons. After covering all the architectures, we will provide a short conclusion.

## 1. Decoders (Llama, Qwen, Gemma, ...)   
Decoders are autoregresive architectures, meaning that they are exclusively trained for the next token prediction given the previous tokens. This inheritly makes them the most suitable for generating new text as a language model, such as answering questions, continuing a story, etc. In our case, the input would be the input text with the target word(s) marked inside "<>". The target would be the emoji sequence or the JSON format including the emoji sequence in its "emoji" field. The loss would be the autoregressive cross-entropy (sum of the negative log probabilities) over the ground-truth emoji tokens.

**Pros of Decoders**     
1. Very easy to fine-tune with existing SFT frameworks, such as TRL's SFTTrainer, which by default only works with the decoder-only models.
2. They can generate variable-length (open-ended) outputs, which matters because one marked word can in theory be translated to an emoji sequence of arbitrary length.
3. It is the most straight-forward and popular solution, which establishes a strong and reasonable baseline.

**Cons of Decoders**  
1. Our task of translating the marked word(s) to an emoji sequence does not align very well with the autoregressive training objective of decoders. They are instead most ideal for free-form continuations, not translating an existing text into another form.
2. Decoder sees the input only as a prefix. The architecture is not designed for bidirectional encoding (only unidirectional) before generation. This limits of its depth of understanding of the input which we are trying to translate into an emoji sequence.

## 2. Encoders (BERT, XLM, ...)  
Encoders translate (encode) each token of an input sequence into their corresponding contextual embeddings. They do so by reading the entire input at once, typically bidirectionally to have a better understanding of the context. At the output, we have a set of embeddings corresponding to each input token. This makes them well-suited for classification tasks as each of the resulting embeddings should be assigned to a class. We can also have a single pooled embedding for a single sentence-level vector (like BERT's [CLS] token), which would allow us to assign a label to the whole sequence. This architecture is hard to use for our task since we want to support variable-length emoji sequences at the output. If only one emoji were to be enough, we could try to classify the pooled embedding as one of the emojis (for example, by selecting the closest emoji in the embedding space).

If we were able to create a vocabulary of possible emoji sequences, we could then project the single pooled output embedding to a vector of size of the vocabulary, take the softmax of it (over the vector entries), and treat it as a probability distribution over the vocabulary. With this, we could again use the cross-entropy loss. However, creating a vocabulary of possible emoji sequences is virtually impossible as we have thousands of emojis and the number of possible sequences grows exponentially.

**Pros of Encoders**     
1. Bidirectional encoding allows for a better and deeper understanding of the input compared to decoders.

**Cons of Encoders**    
1. Cannot naturally generate variable-length emoji sequnces.
2. The usage of it for our task (formulated as a classification task) would likely require a set of predefined emoji sequences, which is not feasible.

## 3. Encoders + Decoders (BART, T5, ...)  
Encoders + decoders first translate each token of an input sequence into their corresponding contextual embeddings, just like encoders, typically using bidirectional encoding. Then, they operate on these translated embeddings to decode (generate) a variable-length output just like a decoder. In summary, they act like a decoder from outside in the sense that they take an input sequence and create an open-ended output sequence based on this previous context. However, internally, they first bidirectionally encode the input into a new set of embeddings for a better understanding. Since they are decoders at the output, we can again use the autoregressive cross-entropy (sum of the negative log probabilities) over the ground-truth emoji tokens.

**Pros of Encoders + Decoders**       
1. Bidirectional encoding allows for a better and deeper understanding of the input compared the decoders.
2. Encoding architecture better aligns with our task than a plain decoder as the task is essentially translation.
3. Autoregressive decoding still allows for variable-length emoji sequences at the output.

**Cons of Encoders + Decoders**  
1. Not easy to fine-tune with existing SFT frameworks, such as TRL's SFTTrainer, which by default only works with the decoder-only (and not seq2seq) models.
2. Introducing an encoder part on top of the decoder part naturally makes the model larger. This might not align very well with our task of developing a "cheap and fast" model that can run on small devices, and might carry the risk of overfitting during training.

## Conclusion  
Decoder-only models (Llama, Qwen, Gemma) seem to be the most practical starting baseline: they support variable-length emoji generation and integrate directly with standard SFT tools. Once a baseline is established, encoder–decoder models (BART, T5) are likely the most promising next step, since their bidirectional encoder is better aligned with the translation-style nature of the task. In contrast, encoder-only models do not seem promising because they cannot generate open-ended emoji sequences and would force the task into an unrealistic classification setup.

