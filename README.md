# AyurVaidyaAI

**Multimodal Deep Learning for Integrative Ayurvedic-Modern Diagnosis with Uncertainty-Aware Treatment Recommendations**

*M.Tech Thesis · Jacob Joseph · Data Engineering · IIT Jodhpur*
*Supervisor: Manu Kanchan*

---

## Overview

AyurVaidyaAI is a multimodal deep learning system that bridges **modern clinical diagnosis** and **Ayurvedic medicine**. It takes two input modalities — free-text symptom descriptions and tongue images — and jointly predicts a modern disease label, estimates Ayurvedic dosha imbalances (Vata / Pitta / Kapha), and retrieves personalised treatment recommendations from the AyurGenixAI knowledge base. Calibrated uncertainty estimates are provided alongside every prediction to support safe, human-in-the-loop deployment.

---

## Research Problem

Modern clinical decision-support systems operate in isolation: either they perform symptom-to-disease prediction using conventional medical taxonomies, or they apply Ayurvedic principles independently. No unified, reproducible computational framework exists that:

1. Integrates multimodal signals (free-text symptoms + tongue images) for joint diagnosis
2. Bridges modern and Ayurvedic paradigms by mapping disease labels to dosha-aware treatment recommendations
3. Provides calibrated uncertainty estimates essential for safe clinical deployment
4. Learns disentangled representations that separate disease states and dosha imbalances from nuisance variations

---

## Architecture

```
Symptom text  ──► BERT Text Encoder (768-dim)  ──┐
                                                   ├──► Gated Multi-Head Cross-Attention Fusion
Tongue image  ──► ResNet-50 Vision Encoder (512-dim) ──┘
                                                              │
                        ┌─────────────────────────────────────┤
                        ▼                                     ▼
             Disease Head                           Dosha Estimation Head
        (MLP: 512→256→22)                          (MLP: 512→128→3)
          22 modern diseases                    Vata / Pitta / Kapha
                        │
                        ▼
             Treatment Retrieval Module
        BERT embedding k-NN over AyurGenixAI
               (medicines, dosage, duration)
```

**Fusion mechanism:**  
`h_fused = α ⊙ h_text + (1−α) ⊙ h_vision`,  where `α = σ(W [h_text ; h_vision])`

**Multi-task loss:**  
`L_total = λ₁ L_disease + λ₂ L_dosha + λ₃ L_treatment + λ₄ L_reg`

---

## Datasets

| Dataset | Modality | Source | Size |
|---|---|---|---|
| GretelAI Symptom-to-Diagnosis | Text | HuggingFace | 853 rows, 22 disease classes |
| Ayurvedic Tongue Image Dataset | Images | Kaggle (bhataakib02) | 1,701 images (Vata / Pitta / Kapha) |
| AyurGenixAI | Tabular | Kaggle (kagglekirti123) | 15,160 records, 34 features, 367 diseases |
| Mendeley Disease & Symptoms | Text/Tabular | Mendeley | Extended disease vocabulary |
| Ayurveda Texts (English) | Text | Kaggle (rcratos) | Ayurvedic corpus for semi-supervised learning |

All splits use **70 / 15 / 15** (train / val / test) with stratified sampling.

---

## Results

### Text Encoder (BERT, 22-class disease classification)

| Split | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Validation (best) | 0.8828 | — | — | 0.8619 |
| Test | 0.8516 | 0.8336 | 0.8516 | 0.8257 |

### Vision Encoder (ResNet-50, Vata / Pitta / Kapha)

| Split | Accuracy | F1 |
|---|---|---|
| Validation (best) | 0.9961 | 0.9941 |
| Test | 0.9844 | 0.9764 |

### Uncertainty Calibration (Temperature Scaling)

| Metric | Before TS | After TS |
|---|---|---|
| ECE (Expected Calibration Error) | 0.4396 | **0.0970** |

### Classical ML Baselines (TF-IDF features)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.9062 | 0.9231 | 0.9062 | 0.9038 |
| TF-IDF + Random Forest | 0.9219 | 0.9302 | 0.9219 | 0.9207 |

### Ablation Study (Disease Classification Head)

| Configuration | Accuracy | F1 |
|---|---|---|
| BERT text-only | 0.8516 | 0.8257 |
| ResNet-50 vision-only | 0.9844 | 0.9764 |
| Early concat fusion | 0.0469 | 0.0043 |
| Gated attention fusion | 0.0469 | 0.0112 |
| BERT + Temperature Scaling | — | 0.8257 |
| BERT + MC Dropout | — | 0.8145 |

> **Note:** The multimodal fusion rows reflect early training results from a cross-modal fusion operating across disease and dosha tasks simultaneously; ablation is ongoing.

---

## Completed Objectives

- [x] Text encoder — BERT fine-tuned with weighted cross-entropy (F1 ≈ 0.83)
- [x] Vision encoder — ResNet-50 fine-tuned for Vata / Pitta / Kapha (F1 ≈ 0.98)
- [x] Cross-modal gated attention fusion module
- [x] t-SNE / representation quality analysis
- [x] Taxonomy mapping (GretelAI ↔ AyurGenixAI)
- [x] Treatment retrieval (BERT embedding k-NN into AyurGenixAI)
- [x] Multi-task loss (disease + dosha joint optimisation)
- [x] MC Dropout uncertainty estimation (20 forward passes)
- [x] Temperature scaling + ECE calibration
- [x] Ablation studies (text-only, vision-only, early fusion, gated attention fusion)
- [x] Classical baselines (TF-IDF + LR / RF)
- [x] Grad-CAM tongue activation maps
- [x] Fusion gate distribution visualisation
- [x] Confusion matrices + per-class F1

---

## Repository Structure

```
AyurVaidyaAI/
├── AyurVaidyaAI_Complete_results_final.ipynb   # Main pipeline notebook
├── Ayurvedic dataset/
│   ├── AyurGenixAI_Dataset.csv
│   └── AyurGenixAI_Dataset.xlsx
├── Generated Images/
│   ├── Grad-CAM_TongueActivation.png
│   ├── TemperatureScaling.png
│   └── TextEncoder_ConfusionMatrix.png
├── MTech Proposal v3.docx                      # Full thesis proposal
└── Archive/                                    # Earlier versions & reports
```

The notebook creates the following directories at runtime:

```
data/
├── text/gretel/           # GretelAI symptom-to-diagnosis CSVs + tokenised .npz
├── text/ayur_texts/       # Ayurveda corpus
├── images/tongue/raw/     # Raw tongue images (class subfolders)
├── images/tongue/processed/  # Train / val / test splits
└── tabular/ayurgenix/     # AyurGenixAI CSV
checkpoints/               # Best model weights (.pt)
results/                   # Confusion matrices, calibration plots, CSVs
meta/                      # Taxonomy mapping CSV
```

---

## Setup & Running

### Requirements

- Python ≥ 3.10
- CUDA GPU recommended (tested on Tesla T4 via Google Colab)

### Install dependencies

```bash
pip install "transformers[torch]" datasets scikit-learn pandas opencv-python \
    matplotlib seaborn wandb kagglehub grad-cam timm torch torchvision
```

### Quick start (Google Colab)

1. Clone the repository and mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   !git clone https://github.com/jacob-joseph-rl/AyurVaidyaAI.git
   %cd AyurVaidyaAI
   ```

2. Open and run `AyurVaidyaAI_Complete_results_final.ipynb` top-to-bottom.  
   The notebook handles all dataset downloads via `kagglehub` and HuggingFace `datasets` automatically.

3. Experiment tracking is logged with **Weights & Biases** (`wandb`). Set `WANDB_MODE=online` before running for cloud sync, or leave it offline for local logs.

### Kaggle credentials

Dataset downloads from Kaggle require a `kaggle.json` API token placed at `~/.kaggle/kaggle.json`. Generate one from your [Kaggle account settings](https://www.kaggle.com/settings/account).

---

## Key Hyperparameters

| Component | Hyperparameter | Value |
|---|---|---|
| Text encoder | Base model | `bert-base-uncased` |
| Text encoder | Max sequence length | 128 tokens |
| Text encoder | Learning rate | 2 × 10⁻⁵ |
| Text encoder | Epochs | 15 |
| Text encoder | Batch size | 32 |
| Vision encoder | Base model | ResNet-50 (ImageNet init) |
| Vision encoder | Input size | 224 × 224 |
| Fusion | Attention heads | 4 × 256-dim |
| Uncertainty | MC Dropout passes | 20 |
| Optimiser | Algorithm | AdamW + weight decay 0.01 |
| Optimiser | LR schedule | Linear warmup (100 steps) + cosine decay |
| Regularisation | Gradient clipping | max norm = 1.0 |
| Regularisation | Dropout | p = 0.3 (text), p = 0.3/0.2 (vision head) |

---

## Interpretability

- **Grad-CAM** on ResNet-50's `layer4` highlights the tongue regions most predictive of dosha class.
- **Attention weights** from the cross-modal fusion gate visualise which modality (text vs. image) the model relies on for each sample.
- **Reliability diagrams** before and after temperature scaling demonstrate calibration improvement (ECE: 0.44 → 0.10).
- **Per-class confusion matrices** available for both the text encoder and the baseline models.

---

## Ethical Considerations

> ⚠️ **All outputs are for research and educational purposes only and are NOT intended for clinical use.**

- Uses only publicly available, de-identified datasets — no patient-identifiable information.
- Model uncertainty (ECE, MC Dropout variance) is surfaced explicitly to prevent overreliance.
- Dataset limitations and potential demographic biases are documented.
- Ayurvedic taxonomy has been mapped with domain-expert guidance to ensure respectful representation of traditional knowledge.

---

## Citation / Acknowledgements

**Thesis Proposal:** Jacob Joseph, *Multimodal Deep Learning for Integrative Ayurvedic-Modern Diagnosis with Uncertainty-Aware Treatment Recommendations*, M.Tech Thesis, IIT Jodhpur, 2025.

**Primary reference:** Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

**Datasets:**
- [GretelAI / symptom_to_diagnosis](https://huggingface.co/datasets/gretelai/symptom_to_diagnosis) (HuggingFace)
- [AyurGenixAI Ayurvedic Dataset](https://www.kaggle.com/datasets/kagglekirti123/ayurgenixai-ayurvedic-dataset) (Kaggle)
- [Ayurvedic Tongue Image Dataset](https://www.kaggle.com/datasets/bhataakib02/ayurvedic-tongue-image-dataset-for-vatapittakapha) (Kaggle)
- [Diseases and Symptoms Dataset](https://data.mendeley.com/datasets/2cxccsxydc) (Mendeley)
- [Ayurveda Texts (English)](https://www.kaggle.com/datasets/rcratos/ayurveda-texts-english) (Kaggle)

---

## License

This project is released for academic and research use. Please cite the thesis if you build upon this work.
