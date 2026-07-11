# 🏥 HealthMate-AI: Medical Question-Answering Chatbot

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Deployed on Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7.svg)](https://render.com)

> **A Retrieval-Augmented Generation (RAG) medical chatbot delivering reliable, grounded medical information through fine-tuned embeddings and LLMs.**
---

## 📋 Table of Contents

- [Overview](#-overview)
- [Published Models](#-published-models-on-hugging-face)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Performance Metrics](#-performance-metrics)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Fine-Tuning Process](#-fine-tuning-process)
- [Deployment](#-deployment)
- [Technologies Used](#-technologies-used)
- [Future Work](#-future-work)
- [License](#-license)

---

## 🎯 Overview

HealthMate-AI is an advanced medical information chatbot that combines:

1. **Custom Fine-Tuned Embeddings** - 3-fold ensemble model achieving **+18.31% improvement** over baseline
2. **Retrieval-Augmented Generation (RAG)** - Grounded responses from medical encyclopedia
3. **Vector Database** - Pinecone for efficient semantic search
4. **Production Deployment** - Live web application on Render

The system retrieves contextually relevant information from *The Gale Encyclopedia of Medicine* and generates accurate, trustworthy responses using state-of-the-art language models.

---

## 🤗 Published Models on Hugging Face

All fine-tuned models are publicly available:

| Model | Description | Link |
|-------|-------------|------|
| **healthmate-mistral-7b-medical-lora** | LoRA adapters for Mistral-7B-Instruct-v0.2, fine-tuned on medical instruction data (3 seeds: 42 at root, 123/999 in subfolders) | [🤗 Hub](https://huggingface.co/yogvidwankhede/healthmate-mistral-7b-medical-lora) |
| **healthmate-minilm-l6-v2-medical-3fold** | Fine-tuned all-MiniLM-L6-v2 embeddings, 3-fold CV ensemble (Spearman 0.8039, +18.31% vs base) | [🤗 Hub](https://huggingface.co/yogvidwankhede/healthmate-minilm-l6-v2-medical-3fold) |
| **healthmate-minilm-l6-v2-medical-best2** | Ensemble of the 2 best CV folds (Spearman 0.8037) | [🤗 Hub](https://huggingface.co/yogvidwankhede/healthmate-minilm-l6-v2-medical-best2) |

```python
# Embeddings
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer("yogvidwankhede/healthmate-minilm-l6-v2-medical-3fold")

# LLM (LoRA adapter on Mistral-7B)
from transformers import AutoModelForCausalLM
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="auto")
model = PeftModel.from_pretrained(base, "yogvidwankhede/healthmate-mistral-7b-medical-lora")
```

---

## ✨ Key Features

### 🧠 Dual Fine-Tuning Approach
- **Custom Embeddings**: Fine-tuned all-MiniLM-L6-v2 with 3-fold cross-validation
- **Language Models**: LoRA fine-tuned Mistral-7B (optional integration)

### 🔍 Advanced Retrieval
- **Vector Database**: 40,000+ medical text chunks in Pinecone
- **Semantic Search**: Cosine similarity with fine-tuned embeddings
- **Smart Chunking**: 500-token chunks with 20-token overlap

### 💬 User-Friendly Interface
- **Dark/Light Mode**: Adaptive UI themes
- **Conversation History**: Sidebar with previous Q&A pairs
- **Real-time Responses**: Streaming answers with loading indicators

### 📊 Rigorous Evaluation
- **BLEU & ROUGE-L Scores**: Quantitative quality metrics
- **535 Test Questions**: Comprehensive medical query dataset
- **Spearman Correlation**: Embedding quality validation (0.8039)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE (Flask)                    │
│                      chat.html + app.py                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   RAG PIPELINE (LangChain)                   │
├─────────────────────────────────────────────────────────────┤
│  1. Query Embedding (Custom Fine-tuned Model)               │
│  2. Vector Search (Pinecone - Top 3 results)                │
│  3. Context Assembly (Retrieved + Prompt Template)          │
│  4. LLM Generation (GPT-4o / Mistral-7B)                    │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        ▼                                  ▼
┌──────────────────┐            ┌──────────────────┐
│ CUSTOM EMBEDDINGS│            │  VECTOR DATABASE │
├──────────────────┤            ├──────────────────┤
│ 3-Fold Ensemble  │            │ Pinecone Index   │
│ Spearman: 0.8039 │            │ 40K chunks       │
│ Dimension: 384   │            │ Medical texts    │
└──────────────────┘            └──────────────────┘
```

### Data Flow

1. **User Input** → Flask web app receives medical query
2. **Embedding** → Query converted to 384-dim vector using fine-tuned model
3. **Retrieval** → Pinecone returns top-3 semantically similar chunks
4. **Generation** → LLM synthesizes answer grounded in retrieved context
5. **Response** → User receives accurate, citation-ready medical information

---

## 📈 Performance Metrics

### Embedding Model Performance

| Model | Spearman Correlation | Improvement vs Base |
|-------|---------------------|---------------------|
| **3-Fold Ensemble** | **0.8039** | **+18.31%** |
| Best 2 Ensemble | 0.8037 | +18.28% |
| Fold 1 | 0.8022 | +18.06% |
| Fold 2 | 0.8017 | +17.98% |
| Fold 3 | 0.8004 | +17.79% |
| Fine-tuned L6V2 | 0.7016 | +3.25% |
| **Base L6V2** | **0.6795** | **baseline** |

### Language Model Quality (RAG System)

**Evaluation on 535 medical questions:**

- **BLEU Score**: [Your actual score]
- **ROUGE-L Score**: [Your actual score]
- **Max Score**: [Your actual score]

*Metrics computed against reference answers from medical experts*

### Key Improvements

✅ **18.31% better semantic understanding** of medical queries  
✅ **Improved cluster consistency** - medical concepts properly grouped  
✅ **Higher IIDR metric** - better embedding space structure  
✅ **Reduced hallucination** - responses grounded in encyclopedia text

---

## 📁 Project Structure

```
HealthMate-AI/
│
├── 📱 DEPLOYMENT (Production)
│   ├── app.py                          # Flask web application
│   ├── chat.html                       # User interface
│   ├── requirements.txt                # Python dependencies
│   └── src/
│       ├── helper.py                   # Custom embeddings loader
│       └── prompt.py                   # System prompts
│
├── 🧠 EMBEDDING FINE-TUNING
│   ├── 3_Fold_CV_L6V2.py              # 3-fold cross-validation training
│   ├── Best_2_Folds_Ensemble_L6V2.py  # Best 2 ensemble creation
│   ├── fine_tune.py                    # Single model fine-tuning
│   ├── test_models.py                  # Model evaluation & comparison
│   └── embeddings.py                   # Custom embedding classes
│
├── 🤖 LLM FINE-TUNING (Optional)
│   ├── config.py                       # Mistral-7B configuration
│   ├── prepare_data.py                 # Dataset preparation
│   ├── finetune_mistral_lora.py       # LoRA fine-tuning script
│   ├── train_multi_adapters.py        # Multi-seed adapter training
│   ├── evaluate_adapters.py           # Adapter evaluation
│   ├── langchain_wrapper.py           # LangChain integration
│   └── local_training.ipynb           # Jupyter training notebook
│
├── 📊 EVALUATION & METRICS
│   ├── eval_metrices.py               # BLEU/ROUGE-L computation
│   ├── medical_questions.csv          # Test dataset (460 questions)
│   └── new_medical_questions.csv      # Extended dataset (535 questions)
│
├── 🔬 RESEARCH & EXPERIMENTATION
│   ├── research.ipynb                 # Main RAG pipeline notebook
│   ├── data_saving.py                 # PDF loading & caching
│   └── data_format.json               # Training data schema
│
├── 📈 VISUALIZATIONS
│   ├── presentation_slide_2_metrics.png          # Model comparison table
│   ├── all_models_comprehensive_comparison.png   # Performance overview
│   ├── llm_similarity_to_reference.png          # LLM quality metrics
│   ├── llm_similarity_mean_ci.png               # Similarity with CI
│   ├── llm_similarity_per_prompt.png            # Per-prompt analysis
│   ├── embedding_cluster_consistency.png        # Cluster validation
│   ├── emb_scatter_tsne.png                    # t-SNE visualization
│   ├── emb_scatter_pca.png                     # PCA visualization
│   ├── emb_iidr_with_ci.png                    # IIDR metric
│   └── emb_cosine_heatmaps_delta.png           # Similarity heatmaps
│
└── 📚 DATA (Not in Git - .gitignored)
    ├── The Gale Encyclopedia.pdf       # 4505-page medical encyclopedia
    └── pdf_cache.pkl                   # Cached extracted documents
```

---

## 🚀 Installation

### Prerequisites

- Python 3.9 - 3.12
- pip package manager
- Git

### 1. Clone Repository

```bash
git clone https://github.com/yogvidwankhede/HealthMate-AI.git
cd HealthMate-AI
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the root directory:

```env
# Required for production deployment
PINECONE_API_KEY=your_pinecone_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
HF_MODEL_NAME=yogvidwankhede/healthmate-minilm-l6-v2-medical-3fold

# Optional for local development
PYTHON_VERSION=3.9.18
```

**Get API Keys:**
- Pinecone: https://www.pinecone.io/
- OpenAI: https://platform.openai.com/api-keys
- Hugging Face (for custom model): https://huggingface.co/settings/tokens

---

## 💻 Usage

### Local Development

#### 1. Test Custom Embeddings

```bash
python test_custom_embeddings.py
```

Expected output:
```
✅ Model loaded from HuggingFace Hub
Embedding dimension: 384
Similarity tests:
  diabetes + treatment signs: 0.9823 (High ✓)
  diabetes + car accident: 0.1580 (Low ✓)
```

#### 2. Run Flask App

```bash
python app.py
```

Visit: http://localhost:8080

#### 3. Run Jupyter Notebook (Full RAG Pipeline)

```bash
jupyter notebook research.ipynb
```

### Production Deployment

See [Deployment](#-deployment) section below.

---

## 🧪 Fine-Tuning Process

### A. Embedding Model Fine-Tuning

#### Step 1: Prepare Medical Similarity Dataset

The dataset contains sentence pairs with similarity scores (0-1):

```csv
sentence1,sentence2,score
"What is diabetes?","What are the symptoms of diabetes?",0.85
"Heart attack causes","Symptoms of myocardial infarction",0.78
"Broken bone treatment","How to treat a fracture",0.92
```

#### Step 2: 3-Fold Cross-Validation Training

```bash
python 3_Fold_CV_L6V2.py
```

**Process:**
1. Splits data into 3 folds
2. Trains separate models on each fold
3. Evaluates on held-out test set
4. Creates ensemble by averaging weights

**Output:**
- `l6v2_3fold_models/l6v2_fold1_best.pt`
- `l6v2_3fold_models/l6v2_fold2_best.pt`
- `l6v2_3fold_models/l6v2_fold3_best.pt`
- `l6v2_3fold_models/l6v2_3fold_ensemble/` (final model)

#### Step 3: Create Best 2 Ensemble (Optional)

```bash
python Best_2_Folds_Ensemble_L6V2.py
```

Identifies top 2 performing folds and averages their weights.

#### Step 4: Model Evaluation

```bash
python test_models.py
```

Generates comprehensive comparison visualizations:
- Bar charts comparing all models
- Category-wise performance breakdown
- Improvement percentages over baseline
- Statistical significance tests

### B. LLM Fine-Tuning (Optional - Mistral-7B)

#### Step 1: Prepare Training Data

```bash
python prepare_data.py
```

Converts medical Q&A CSV to instruction format:

```json
{
  "instruction": "What is pneumonia?",
  "context": "Medical Information: Pneumonia is an infection...",
  "response": "Pneumonia is an infection that inflames the air sacs..."
}
```

#### Step 2: Configure Training

Edit `config.py`:

```python
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
LORA_CONFIG = {
    "r": 16,                  # LoRA rank
    "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj", ...]
}
TRAINING_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "learning_rate": 2e-4,
}
```

#### Step 3: Train LoRA Adapters

**Single adapter:**
```bash
python finetune_mistral_lora.py
```

**Multiple seeds (ensemble approach):**
```bash
python train_multi_adapters.py
```

Trains 3 adapters with seeds [42, 123, 999] for robustness.

#### Step 4: Evaluate Adapters

```bash
python evaluate_adapters.py
```

Computes BLEU/ROUGE-L scores for each adapter.

#### Step 5: Use in RAG Pipeline

```python
from langchain_wrapper import create_medical_llm

# Load fine-tuned Mistral instead of GPT-4o
chatModel = create_medical_llm(
    adapter_path="./mistral_medical_lora_models/adapter_seed_42"
)
```

---

## 🌐 Deployment

### Deploy to Render (Free Tier)

#### 1. Push to GitHub

```bash
git add .
git commit -m "Ready for deployment"
git push origin main
```

#### 2. Create Render Web Service

1. Go to https://render.com/
2. Click **New** → **Web Service**
3. Connect your GitHub repository
4. Configure:

**Settings:**
```
Name: healthmate-ai
Environment: Python 3
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 1 --timeout 120
```

**Environment Variables:**
```
PINECONE_API_KEY = pc_xxxxx...
OPENAI_API_KEY = sk-xxxxx...
HF_MODEL_NAME = yogvidwankhede/healthmate-minilm-l6-v2-medical-3fold
PYTHON_VERSION = 3.9.18
WEB_CONCURRENCY = 1
TOKENIZERS_PARALLELISM = false
OMP_NUM_THREADS = 1
```

#### 3. Deploy

Click **Create Web Service**. Build takes 3-5 minutes.

#### 4. Access Your App

```
https://healthmate-ai.onrender.com
```

### Memory Optimization (Free Tier - 512MB)

If deployment fails with "Out of memory":

**Option A: Optimize (Free)**
- Workers set to 1
- Environment variables added
- Model lazy-loaded

**Option B: Upgrade ($7-25/month)**
- Starter: 1GB RAM - sufficient
- Standard: 2GB RAM - recommended

---

---

## 🛠️ Technologies Used

### Core Framework
- **LangChain** 0.3.26 - RAG orchestration
- **Flask** 3.1.0 - Web application
- **Gunicorn** 23.0.0 - Production WSGI server

### Machine Learning
- **PyTorch** 2.5.1 - Deep learning framework
- **Transformers** 4.37.2 - Hugging Face models
- **Sentence-Transformers** 4.1.0 - Embedding models
- **PEFT** 0.7.1 - Parameter-efficient fine-tuning (LoRA)

### Vector Database
- **Pinecone** 3.0.0 - Vector storage & search
- **LangChain-Pinecone** 0.2.0 - Integration

### Language Models
- **OpenAI GPT-4o** - Primary generation (via API)
- **Mistral-7B-Instruct** - Optional fine-tuned model

### Data Processing
- **LangChain-Community** 0.3.13 - Document loaders
- **PyPDF** 3.17.1 - PDF extraction
- **Pandas** 2.2.3 - Data manipulation
- **Scikit-learn** 1.6.1 - Cross-validation

### Evaluation
- **NLTK** - BLEU score computation
- Custom ROUGE-L implementation
- Spearman correlation analysis

### Visualization
- **Matplotlib** 3.10.0 - Plotting
- **Seaborn** 0.13.2 - Statistical graphics

---

## 🔮 Future Work

### Short-term Improvements
- [ ] Add medical specialty filters (cardiology, neurology, etc.)
- [ ] Implement conversation memory across sessions
- [ ] Add source citation display in UI
- [ ] Multi-language support (Spanish, Mandarin)

### Advanced Features
- [ ] **Voice Interface** - Audio input/output
- [ ] **Image Analysis** - Medical image interpretation
- [ ] **Symptom Checker** - Interactive diagnosis assistant
- [ ] **Drug Interaction Checker** - Safety verification

### Model Enhancements
- [ ] Fine-tune larger embedding models (768-dim)
- [ ] Ensemble multiple LLMs (GPT-4o + Mistral + Llama)
- [ ] Add specialized medical LLMs (BioGPT, PubMedBERT)
- [ ] Implement active learning from user feedback

### Infrastructure
- [ ] Redis caching for frequent queries
- [ ] PostgreSQL for conversation logging
- [ ] Prometheus monitoring & alerting
- [ ] A/B testing framework

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **The Gale Encyclopedia of Medicine** - Medical knowledge base
- **Hugging Face** - Model hosting & transformers library
- **Pinecone** - Vector database platform
- **OpenAI** - GPT-4o API access

---

---

## 📚 Citations

If you use this work in your research, please cite:

```bibtex
@misc{wankhede2024healthmate,
  title={HealthMate-AI: A Retrieval-Augmented Generation Medical Chatbot with Fine-Tuned Embeddings},
  author={Wankhede, Yogvid and Nan, Leonardo},
  year={2024},
  institution={Washington University in St. Louis},
  course={ESE 5971 - Practicum in Data Analytics and Statistics}
}
```

</div>
