# Proteomics_analysis_AAA

**Plasma Proteomics Profiling for Abdominal Aortic Aneurysm (AAA) Biomarker Discovery**  
A comprehensive data analysis pipeline using statistical analysis, pathway enrichment, co-expression networks, and machine learning models.

---

## 🧬 Overview

This repository contains the full pipeline used for the analysis of plasma proteomics data from AAA patients and controls, as part of a biomarker discovery study. Using mass spectrometry data, the study explores differential protein abundance, pathway enrichment, network analysis, and machine learning–based prediction to uncover novel diagnostic and prognostic biomarkers of AAA.

---

## 📁 Project Structure
```
Proteomics_analysis_AAA/
│
├── README.md ← This file
│
├── Data/
│ └── gsea_gmt.gmt ← Custom gene set for GSEA analysis
│
├── Module/
│ ├── DAP.py ← Python module for proteomics data analysis
│ ├── mkdocs.yml ← MkDocs configuration for documentation
│ ├── docs/
│ │ └── index.md ← Documentation source (Markdown)
│ └── site/ ← Auto-generated MkDocs static site
│ └── ... ← Static assets (HTML, JS, CSS, etc.)
│
└── Notebooks/
├── BINN.ipynb ← Biologically informed neural network experiments
├── DAP_AAA_CONTROL.ipynb ← Quality control and DAP analysis between AAA and control samples
├── GSEA.ipynb ← Gene Set Enrichment Analysis
├── meta_EDA.ipynb ← Metadata exploration
├── ML.ipynb ← SVM-based predictive modeling
└── network.ipynb ← Co-expression network analysis
├── Other/
│ └── requirements.txt ← Required packages for the module
```
---

## 🧪 Methodology Highlights

- **Mass Spectrometry Data**: Plasma samples processed with DIA-PASEF on TimsTOF HT.
- **Differentially Abundant Proteins (DAPs)**: Identified with p-value & fold-change thresholds, using t-tests and non-parametric alternatives.
- **GSEA**: Enrichment analysis using curated pathways and the `GSEAPY` library.
- **Network Analysis**: Co-expression networks constructed via Spearman correlations.
- **Machine Learning**: SVMs with nested cross-validation; ROC AUC scores up to 0.94.
- **Model Interpretability**: Feature importance via RFE and SVM decision boundaries.

---

## ⚙️ Installation & Requirements

In order to use the analysis module, simply download the DAP.py file.
Documentation can be found at: https://kedijkstra.github.io/Proteomics_analysis_AAA/

### Install required packages
pip install -r requirements.txt
⚠️ Note: VSN has to be downloaded manually from https://github.com/stmball/vsn
