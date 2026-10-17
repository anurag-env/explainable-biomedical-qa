# RAG-RERoBERTa-XAI-TTS

**Recursive Attention-Guided RoBERTa Framework for Explainable Multilingual Biomedical Question Answering and Voice-Based Clinical Decision Support**

A biomedical QA system that combines a RoBERTa encoder, a novel **recursive embedding refinement** mechanism, graph-attention evidence aggregation, explainability (SHAP / Integrated Gradients), English↔Hindi translation, and multilingual text-to-speech — producing answers that are accurate, explainable, and spoken in the user's language.

This README is written as a **working brief for Antigravity agents**: each section below maps to one agent task/workspace. Paste the relevant section (or the whole file) into the Antigravity Agent Manager as your task prompt, and let the agent produce an implementation plan before it writes code.

---

## 1. Project Goal

Build a biomedical QA pipeline that:

- Answers biomedical questions with `Y ∈ {Yes, No, Maybe}` or free-text answers
- Iteratively refines its internal representation of the question/evidence until convergence
- Grounds every answer in retrieved PubMed evidence via a knowledge graph
- Explains *why* it gave that answer (attention heatmaps, SHAP, Integrated Gradients)
- Estimates a confidence score for each answer
- Translates the answer into Hindi (and back)
- Speaks the answer aloud in English and Hindi

Target outcome: **+4–8% accuracy over BioBERT / PubMedBERT / ClinicalBERT** baselines, submittable to a Scopus Q1 / SCI venue.

---

## 2. Repository Structure

Ask the agent to scaffold this structure first, before writing any model code:

```
.
├── README.md
├── pyproject.toml / requirements.txt
├── configs/
│   ├── data.yaml
│   ├── model.yaml
│   └── train.yaml
├── data/
│   ├── raw/                # untouched downloads
│   ├── processed/          # tokenized, split, cleaned
│   └── scripts/            # download_medqa.py, download_pubmedqa.py, download_healthsearchqa.py
├── src/
│   ├── data/                # Dataset + DataLoader classes
│   ├── models/
│   │   ├── encoder.py       # RoBERTa biomedical encoder
│   │   ├── attention.py     # multi-head self-attention (8/12/16 heads)
│   │   ├── recursive_refine.py   # E_{t+1} = Attention(E_t) + GRU(E_t)
│   │   ├── evidence_graph.py     # graph construction + GAT layer
│   │   ├── explainability.py     # SHAP, Integrated Gradients, heatmaps
│   │   ├── confidence.py         # confidence calibration head
│   │   └── rerb_qa.py             # full model assembling the above
│   ├── translation/          # IndicTrans2 / NLLB-200 / mBART50 wrappers
│   ├── tts/                  # VITS/FastSpeech2/XTTS-v2/Bark + Hindi TTS wrappers
│   ├── train.py               # staged training entrypoint
│   ├── evaluate.py            # QA / explainability / translation / speech metrics
│   └── inference_pipeline.py  # end-to-end: question -> spoken, explained answer
├── notebooks/                 # exploratory analysis, ablations
├── tests/                     # unit tests per module
└── outputs/
    ├── checkpoints/
    ├── logs/
    └── reports/                # generated eval reports, figures
```

---

## 3. Datasets

| Dataset | Samples | Source |
|---|---|---|
| PubMedQA | 273,500 (211,300 artificial + 61,200 unlabeled + 1,000 expert-labeled) | Jin et al., EMNLP-IJCNLP 2019, `pubmed_qa` on Hugging Face |
| MedQA | 12,723 | Jin et al. 2020, [github.com/jind11/MedQA](https://github.com/jind11/MedQA) |
| HealthSearchQA | 4,440 | Singhal et al., *Nature* 2023, `katielink/healthsearchqa` on Hugging Face |

**Agent task:** write `data/scripts/download_*.py` for each source, verify sample counts match the table above, and normalize all three into a common schema: `{question, context/abstract, answer, label}`.

---

## 4. Mathematical Specification (give this verbatim to the modeling agent)

**Input**
```
Question:        Q = {q1, q2, ..., qn}
PubMed Abstract:  A = {a1, a2, ..., am}
Knowledge Context: K = {k1, k2, ..., kl}
```

**Objective:** learn `f(Q, A, K) → Y` where `Y ∈ {Yes, No, Maybe}` or `Y = Textual Answer`

**Loss (minimize):**
```
L = L_qa + λ1 * L_translation + λ2 * L_explainability + λ3 * L_speech
```

**Recursive embedding refinement (the novel component):**
```
E0 = RoBERTa(Q, A)
E_{t+1} = Attention(E_t) + GRU(E_t)
E* = E_n, iterate until ||E_{t+1} - E_t|| < epsilon
```

**Multi-head attention:**
```
Attention(Q, K, V) = Softmax(QK^T / sqrt(d_k)) V
```
Ablate over 8 / 12 / 16 heads.

**Evidence graph:** `G = (V, E)`
- Nodes: Disease, Drug, Symptom, Procedure
- Edges: causes, treats, associated-with
- Aggregation via Graph Attention Network: `h_i' = α_ij W h_j`

**Explainability:**
- Attention heatmaps `H = {h1, h2, ..., hn}` over diseases/symptoms/treatments/biomarkers
- `SHAP(x_i)` for feature importance
- `IG(x)` (Integrated Gradients) for prediction-path explanation

---

## 5. Architecture Pipeline

```
Input Question
  -> Biomedical Preprocessing
  -> RoBERTa Biomedical Encoder
  -> Multi-head Self-Attention
  -> Recursive Embedding Refinement Layer
  -> [Evidence Graph]  +  [Confidence Layer]
  -> Explainable Answer Generator
  -> [Hindi Translator]  +  [English Output]
  -> Explainable Voice Engine
```

Voice output template:
```
Answer: <text>
Confidence: <calibrated %>
Evidence Source: <PubMed ID>
Citation: <formatted citation>
```
Example: *"Based on PubMed abstract 32897211, evidence suggests aspirin reduces risk. Confidence = 92%."*

---

## 6. Staged Training Curriculum

| Stage | Task | Data |
|---|---|---|
| 1 | PubMedQA pretraining | 273,500 samples |
| 2 | MedQA fine-tuning | 12,723 samples |
| 3 | HealthSearchQA fine-tuning | 4,440 samples |
| 4 | Hindi translation fine-tuning | IndicTrans2 |
| 5 | Voice generation training | Hindi + English corpora |

**Agent task:** implement `train.py --stage {1..5}` so each stage can be run, checkpointed, and resumed independently — don't couple all five into one monolithic script.

---

## 7. Hyperparameter Search Space

| Parameter | Range |
|---|---|
| Learning Rate | 1e-5 – 5e-5 |
| Batch Size | 8, 16, 32 |
| Epochs | 5–30 |
| Max Length | 256, 512 |
| Attention Heads | 8, 12, 16 |
| Hidden Units | 768, 1024 |
| Dropout | 0.1–0.5 |
| Weight Decay | 0.01–0.1 |
| Beam Width | 3–10 |
| Recursive Iterations | 3–10 |
| Optimizers | AdamW, Ranger, Lion, Lookahead Adam |

> Use random search or Bayesian optimization (e.g. Optuna) — the full grid is too large to brute-force. Have the agent set up an Optuna study in `src/train.py` rather than nested loops.

---

## 8. Models to Integrate (don't train from scratch — start from pretrained checkpoints)

**Translation (English ↔ Hindi):**
- IndicTrans2 (primary)
- NLLB-200 (baseline)
- mBART50 (baseline)

**Text-to-Speech, English:** VITS, FastSpeech2, XTTS-v2, Bark
**Text-to-Speech, Hindi:** Indic-TTS, Vakyansh, Coqui XTTS, Hindi VITS

---

## 9. Evaluation Metrics

| Category | Metrics |
|---|---|
| QA | Accuracy, F1, Precision, Recall, MCC |
| Explainability | Faithfulness, Sufficiency, Comprehensiveness |
| Translation | BLEU, METEOR, TER, BERTScore |
| Speech | MOS, PESQ, STOI |

Baselines to beat: **BioBERT, PubMedBERT, ClinicalBERT, vanilla RoBERTa**, and modern RAG baselines.

---

## 10. Recommended Antigravity Workflow

Antigravity works best when you scope each agent task narrowly and review its plan before it executes. Suggested task breakdown (one Antigravity workspace/task each):

1. **Scaffold task** — "Create the repo structure in Section 2, with empty modules and a `requirements.txt` covering transformers, torch, torch-geometric, shap, captum, indic-nlp-library, TTS libraries."
2. **Data task** — "Implement `data/scripts/download_*.py` and `src/data/` per Section 3. Verify sample counts."
3. **Baseline task** — "Fine-tune BioBERT/PubMedBERT/ClinicalBERT/RoBERTa on PubMedQA and MedQA; report metrics from Section 9 QA row into `outputs/reports/baselines.md`."
4. **Core model task** — "Implement `src/models/encoder.py`, `attention.py`, `recursive_refine.py` per the math in Section 4. Add unit tests verifying the refinement loop converges (`||E_{t+1}-E_t|| < epsilon`) on synthetic input."
5. **Evidence + explainability task** — "Implement `evidence_graph.py` (GAT) and `explainability.py` (SHAP, Integrated Gradients, heatmap generation) per Section 4."
6. **Translation + TTS task** — "Implement `src/translation/` and `src/tts/` wrappers per Section 8, with a unified interface `translate(text, direction)` and `speak(text, lang)`."
7. **Training task** — "Implement `train.py` with the 5-stage curriculum (Section 6) and an Optuna sweep over Section 7's search space."
8. **Evaluation task** — "Implement `evaluate.py` computing all Section 9 metrics and writing an ablation report (attention heads, recursive iterations) to `outputs/reports/`."

For each task, let the agent produce its **Implementation Plan** and **Task List** first (Antigravity does this automatically) — review it before approving execution, especially for Task 4 (the novel component) where correctness of the math matters most.

---

## 11. Definition of Done

- [ ] All three datasets downloaded, cleaned, and split
- [ ] Baselines (BioBERT/PubMedBERT/ClinicalBERT/RoBERTa) reproduced with logged metrics
- [ ] Recursive embedding refinement implemented and unit-tested for convergence
- [ ] Evidence graph + GAT producing evidence-grounded answers
- [ ] SHAP + Integrated Gradients explanations generated per prediction
- [ ] Confidence calibration layer implemented and evaluated
- [ ] English↔Hindi translation integrated and evaluated (BLEU/METEOR/TER/BERTScore)
- [ ] English + Hindi TTS integrated and evaluated (MOS/PESQ/STOI)
- [ ] Full ablation study (attention heads, recursive iterations) complete
- [ ] Accuracy gain over BioBERT baseline measured and documented
- [ ] Final report/paper draft assembled from `outputs/reports/`

---

## References

- D. Jin, E. Pan, N. Oufattole, W.-H. Weng, H. Fang, and P. Szolovits, "What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams," 2020. https://github.com/jind11/MedQA
- K. Singhal et al., "Large language models encode clinical knowledge," *Nature*, vol. 620, no. 7972, pp. 172–180, Aug. 2023. doi: 10.1038/s41586-023-06291-2. Dataset: https://huggingface.co/datasets/katielink/healthsearchqa
- Q. Jin, B. Dhingra, Z. Liu, W. W. Cohen, and X. Lu, "PubMedQA: A Dataset for Biomedical Research Question Answering," EMNLP-IJCNLP 2019, pp. 2567–2577. doi: 10.18653/v1/d19-1259
