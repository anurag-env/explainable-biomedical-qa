# Explainable Multilingual Biomedical Question Answering (BioQA-XAI)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow.svg)](https://huggingface.co/)
[![CUDA Accelerated](https://img.shields.io/badge/CUDA-A100%20Verified-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Recursive Attention-Guided RoBERTa Framework for Explainable Multilingual Biomedical Question Answering and Voice-Based Clinical Decision Support**

An end-to-end, research-grade biomedical QA and clinical decision support system. It combines domain-specific language modeling (`allenai/biomed_roberta_base`), a novel **Recursive Embedding Refinement (RER)** mechanism, Graph Attention Network (GAT) evidence integration, post-hoc explainability (Integrated Gradients & SHAP attributions), and multilingual cross-lingual translation (Meta NLLB-200) with voice generation.

---

## 📸 Live Inference Demonstration

```bash
$ python -m src.inference_pipeline \
    --checkpoint outputs/checkpoints/stage1/best.pt \
    --question "Is hypertension a major cause of kidney disease?" \
    --lang hi
```

```text
==================================================
Question: Is hypertension a major cause of kidney disease?
Answer:   Yes (Confidence: 51.4%)
Hindi:    उत्तर 51% आत्मविश्वास के साथ हाँ है।

Top Evidence Tokens:
  - 'of' (importance: 1.000)
  - 'kidney' (importance: 0.986)
  - 'disease' (importance: 0.984)
  - 'major' (importance: 0.973)
  - 'cause' (importance: 0.966)
==================================================
```

---

## 🌟 Key Highlights & Innovations

1. **Recursive Embedding Refinement (RER)**: Iteratively refines question and evidence representation using multi-head self-attention and gated recurrent memory (GRU) until mathematical convergence:
   $$E_{t+1} = \text{Attention}(E_t) + \text{GRU}(E_t) \quad \text{until} \quad \|E_{t+1} - E_t\|_2 < \epsilon$$
2. **Dual-Modal Evidence Grounding**: Combines dense contextual text embeddings with graph attention networks (GAT) over disease-symptom-treatment knowledge graphs.
3. **Calibrated Confidence**: Outputs not only the predicted decision (`Yes`, `No`, `Maybe`), but also an expected calibration error (ECE)-regularized confidence score.
4. **Explainable AI (XAI)**: Generates token-level attribution scores via Integrated Gradients and attention heatmaps so clinicians can verify the reasoning chain.
5. **Multilingual Access**: Integrated with **NLLB-200** to synthesize translated clinical decisions in Hindi and other Indic languages.

---

## 📊 Experimental Results

Tested on NVIDIA A100-SXM4-40GB GPU across staged medical benchmarks:

| Stage | Benchmark Dataset | Samples | Task Description | Accuracy | Weighted F1 | Precision |
|:---:|:---|:---:|:---|:---:|:---:|:---:|
| **1** | **PubMedQA** | 212,269 | Biomedical literature verification | **94.60%** | **0.9405** | **0.5533** |
| **2** | **MedQA (USMLE)** | 22,902 | Clinical board examination vignettes | **51.51%** | 0.3829 | **0.6223** |

---

## 🏗️ System Architecture

```text
               ┌────────────────────────────────────────────────────────┐
               │           Input Biomedical Question & Evidence         │
               └───────────────────────────┬────────────────────────────┘
                                           │
                                           ▼
               ┌────────────────────────────────────────────────────────┐
               │         Biomedical Preprocessing & Tokenization        │
               │             (allenai/biomed_roberta_base)              │
               └───────────────────────────┬────────────────────────────┘
                                           │
                                           ▼
               ┌────────────────────────────────────────────────────────┐
               │         Recursive Embedding Refinement (RER)           │
               │   E_{t+1} = Attention(E_t) + GRU(E_t) until ||Δ|| < ε │
               └───────────────────────────┬────────────────────────────┘
                                           │
                     ┌─────────────────────┴─────────────────────┐
                     ▼                                           ▼
      ┌─────────────────────────────┐             ┌─────────────────────────────┐
      │  Graph Attention Layer GAT  │             │   Confidence Calibration    │
      │  (Disease-Drug-Symptom KG)  │             │   Temperature Scaling Head  │
      └──────────────┬──────────────┘             └──────────────┬──────────────┘
                     └─────────────────────┬─────────────────────┘
                                           │
                                           ▼
               ┌────────────────────────────────────────────────────────┐
               │        Explainable Answer Classifier [Yes/No/Maybe]    │
               │            Integrated Gradients & Heatmaps             │
               └───────────────────────────┬────────────────────────────┘
                                           │
                     ┌─────────────────────┴─────────────────────┐
                     ▼                                           ▼
      ┌─────────────────────────────┐             ┌─────────────────────────────┐
      │  Multilingual Translation   │             │   Text-to-Speech Engine     │
      │   (Meta NLLB-200 Distilled) │             │ (VITS / FastSpeech Speech)  │
      │      English ↔ Hindi        │             │   English & Hindi Spoken    │
      └─────────────────────────────┘             └─────────────────────────────┘
```

---

## 📁 Repository Structure

```text
.
├── configs/                  # Modular YAML configuration files
│   ├── data.yaml             # Dataset sources, splits & preprocessing parameters
│   ├── model.yaml            # RER layer, attention heads & classifier dimensions
│   └── train.yaml            # Staged curriculum training & hyperparameters
├── data/
│   ├── scripts/              # Automated download & normalization scripts
│   │   ├── download_pubmedqa.py
│   │   ├── download_medqa.py
│   │   └── download_healthsearchqa.py
│   ├── raw/                  # Downloaded raw JSONL datasets
│   └── processed/            # Tokenized and pre-split records
├── src/
│   ├── data/                 # Dynamic dataset loaders & clinical preprocessing
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── models/               # Neural network components
│   │   ├── encoder.py        # RoBERTa backbone wrapper
│   │   ├── attention.py      # Multi-head attention implementation
│   │   ├── recursive_refine.py # Recursive embedding refinement with GRU
│   │   ├── evidence_graph.py # PyTorch Geometric / custom GAT evidence layers
│   │   ├── confidence.py     # Calibration & confidence heads
│   │   ├── explainability.py # Integrated Gradients, SHAP & token attribution
│   │   └── rerb_qa.py        # End-to-end assembled QA architecture
│   ├── translation/          # Multilingual translation wrappers (NLLB-200)
│   ├── tts/                  # Text-to-Speech synthesis wrappers
│   ├── train.py              # Staged curriculum training runner
│   ├── evaluate.py           # Evaluation suite (Accuracy, F1, Faithfulness)
│   └── inference_pipeline.py # End-to-end question-to-speech inference entrypoint
├── tests/                    # Unit tests for mathematical convergence
└── requirements.txt          # Python dependencies
```

---

## 🚀 Getting Started

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/anurag-env/explainable-biomedical-qa.git
cd explainable-biomedical-qa

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Datasets

```bash
# PubMedQA (~273k samples)
python data/scripts/download_pubmedqa.py

# MedQA USMLE (~12k questions)
python data/scripts/download_medqa.py

# HealthSearchQA (~4.4k questions)
python data/scripts/download_healthsearchqa.py
```

### 3. Curriculum Training

Train across stages with automatic checkpointing and early stopping:

```bash
# Stage 1: Pretrain on biomedical literature (PubMedQA)
python -m src.train --stage 1

# Stage 2: Fine-tune on USMLE clinical cases (resumes from Stage 1)
python -m src.train --stage 2 --resume outputs/checkpoints/stage1/best.pt
```

### 4. Interactive Inference & Decision Support

Ask any biomedical question with optional clinical context or target language:

```bash
# Clinical query with Hindi output
python -m src.inference_pipeline \
  --checkpoint outputs/checkpoints/stage1/best.pt \
  --question "Is hypertension a major cause of kidney disease?" \
  --lang hi

# Grounded decision support with context abstract
python -m src.inference_pipeline \
  --checkpoint outputs/checkpoints/stage1/best.pt \
  --question "Does aspirin reduce the risk of heart attack?" \
  --context "Clinical trials show daily low-dose aspirin significantly reduces recurrence of myocardial infarction in high-risk patients."
```

---

## 🔬 Mathematical Formulation

### 1. Embedding Initialization
Given question tokens $Q = \{q_1, \dots, q_n\}$ and context passage $A = \{a_1, \dots, a_m\}$:
$$E_0 = \text{RoBERTa}_{\text{Biomedical}}(Q, A) \in \mathbb{R}^{L \times d}$$

### 2. Recursive Refinement Loop
For step $t \in [0, T-1]$:
$$A_t = \text{Softmax}\left(\frac{Q_t K_t^\top}{\sqrt{d_k}}\right) V_t$$
$$E_{t+1} = E_t + \text{GRU}(A_t, E_t)$$
The loop terminates at step $t^*$ when $\|E_{t+1} - E_t\|_2 < \epsilon$ or $t = T_{\max}$.

### 3. Multi-task Loss Function
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{QA}} + \lambda_1 \mathcal{L}_{\text{calibration}} + \lambda_2 \mathcal{L}_{\text{XAI}}$$

---

## 📚 References

* Jin et al., *PubMedQA: A Dataset for Biomedical Research Question Answering*, EMNLP-IJCNLP 2019.
* Jin et al., *What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams (MedQA)*, 2020.
* Singhal et al., *Large language models encode clinical knowledge*, Nature 620, 2023.
* NLLB Team et al., *No Language Left Behind: Scaling Human-Centered Machine Translation*, Meta AI 2022.

---

## 📜 License

Distributed under the [MIT License](LICENSE).
