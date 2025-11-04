# English ↔ Darija Translator 🇲🇦

A bidirectional neural machine translation model for translating between English and Moroccan Darija (Moroccan Arabic dialect), built using state-of-the-art transformer architecture.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Project Overview

This project aims to bridge the communication gap between English and Moroccan Darija by providing an accurate, bidirectional translation system. The model is fine-tuned on the **facebook/nllb-200-distilled-600M** multilingual transformer, specifically adapted for the Darija dialect.

### Use Cases
- 💬 **Chatbots** - Enable multilingual customer support
- 📚 **Educational Applications** - Language learning tools
- 🌍 **Cross-cultural Communication** - Breaking language barriers
- 📱 **Mobile Applications** - Real-time translation services

## ✨ Features

- **Bidirectional Translation**: English → Darija and Darija → English
- **State-of-the-art Model**: Fine-tuned NLLB-200 (600M parameters)
- **High Performance**: BLEU score ≥ 25-30 on test corpus
- **Robust Preprocessing**: Advanced text normalization and tokenization
- **Easy to Use**: Simple API for integration

## 🏗️ Architecture

```
Corpus (English ↔ Darija)
           ↓
Preprocessing (Normalization, Tokenization)
           ↓
   ┌──────────────────┐
   │  NLLB Transformer │
   │  Encoder-Decoder  │
   └──────────────────┘
           ↓
    Translation Output
```

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **BLEU** | ≥ 25-30 |
| **METEOR** | High semantic alignment |
| **chrF** | Character-level accuracy |
| **Evaluation Loss** | 1.54 |

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
PyTorch 2.0+
CUDA-compatible GPU (recommended)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/Mohammed-El-Kassoiri/English-to-Darija-Translator.git
cd english-darija-translator

# Install dependencies
pip install -r requirements.txt
```

### Usage

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the model
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Set language codes
tokenizer.src_lang = "eng_Latn"
tokenizer.tgt_lang = "ary_Arab"

# Translate
def translate_english_to_darija(text):
    inputs = tokenizer(text, return_tensors="pt")
    translated_tokens = model.generate(
        **inputs, 
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("__ary_Arab__"),
        max_length=128
    )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

# Example
english_text = "How are you?"
darija_translation = translate_english_to_darija(english_text)
print(f"English: {english_text}")
print(f"Darija: {darija_translation}")
```

## 📁 Project Structure

```
english_darija_translator/
├── data/
│   ├── raw/              # Original dataset
│   └── processed/        # Preprocessed data
├── notebooks/            # Jupyter notebooks
│   └── English_to_Darija_translator.ipynb
├── src/
│   ├── preprocessing.py  # Data preprocessing utilities
│   ├── train.py         # Training script
│   ├── evaluate.py      # Evaluation metrics
│   └── inference.py     # Inference utilities
├── models/              # Saved model checkpoints
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker configuration
└── README.md           # This file
```

## 🔧 Training

### Dataset

- **Size**: ~70,000 parallel sentence pairs
- **Split**: 80% train, 10% validation, 10% test
- **Format**: CSV with `english` and `darija` columns

### Training Configuration

```python
training_args = {
    "num_train_epochs": 5,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_steps": 500,
}
```

### Training Script

```bash
python src/train.py \
    --model_name facebook/nllb-200-distilled-600M \
    --data_path data/processed/dataset.csv \
    --output_dir models/nllb-fine-tuned \
    --num_epochs 5
```

## 📈 Evaluation

The model is evaluated using multiple metrics:

- **BLEU**: Measures translation quality
- **METEOR**: Semantic similarity
- **chrF**: Character n-gram F-score
- **Qualitative Analysis**: Native speaker review

```bash
python src/evaluate.py --model_path models/nllb-fine-tuned --test_data data/processed/test.csv
```

## 🛠️ Technologies Used

- **Python** - Core programming language
- **PyTorch** - Deep learning framework
- **Hugging Face Transformers** - Pre-trained models and tokenizers
- **FastAPI** - API deployment
- **SentencePiece** - Subword tokenization
- **scikit-learn** - Data splitting and preprocessing

## 📝 Methodology (CRISP-DM)

| Phase | Description |
|-------|-------------|
| **Business Understanding** | Define translation objectives and success metrics |
| **Data Understanding** | Collect and explore bilingual corpus |
| **Data Preparation** | Clean, normalize, and tokenize text data |
| **Modeling** | Fine-tune NLLB-200 on Darija dataset |
| **Evaluation** | Calculate BLEU, METEOR, and chrF scores |
| **Deployment** | Create FastAPI endpoint for production |

## 🎯 Roadmap

- [x] Data collection and preprocessing
- [x] Model fine-tuning
- [x] Evaluation on test set
- [ ] FastAPI deployment
- [ ] Docker containerization
- [ ] Web interface development
- [ ] Mobile app integration
- [ ] Continuous model improvement

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- MOHAMMED EL KASSOIRI - [GitHub Profile](https://github.com/Mohammed-El-Kassoiri)

## 🙏 Acknowledgments

- **Meta AI** for the NLLB-200 model
- **Hugging Face** for the Transformers library
- The Moroccan NLP community for dataset contributions

## 📊 Citation

If you use this project in your research, please cite:

```bibtex
@misc{english-darija-translator,
  author = {MOHAMMED EL KASSOIRI},
  title = {English to Darija Translator},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/Mohammed-El-Kassoiri/English-to-Darija-Translator}
}
```

---

⭐ If you find this project useful, please consider giving it a star on GitHub!
