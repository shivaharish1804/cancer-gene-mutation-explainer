# 🧬 Cancer Gene Mutation Explainer

An AI-powered web application that predicts whether a gene mutation is **harmful (pathogenic)** or **harmless (benign)** — and explains the reasoning in plain English using SHAP (Explainable AI).

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🎯 What This Project Does

Genetic mutations can either cause cancer or be completely harmless. Biologists and clinicians spend significant time classifying these variants. This project uses **Machine Learning** to automate that prediction and — crucially — explains *why* the model made each decision using **SHAP (SHapley Additive exPlanations)**.

**Input:** Features of a gene mutation (damage scores, allele frequency, variant type)  
**Output:** Pathogenic / Benign prediction + confidence score + plain-English explanation + SHAP chart

---

## 🚀 Live Demo

▶️ Try it on Streamlit Cloud → https://cancer-gene-mutation-explainer-7kveaj7kh7zcr45kq3oh9t.streamlit.app

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🎯 **Prediction** | Random Forest model trained on ClinVar public data |
| 🧠 **Explainable AI** | SHAP values show which features drove the prediction |
| 🗣️ **Plain English** | Auto-generated reasoning in non-technical language |
| 📊 **Visualisations** | Per-prediction SHAP chart + global feature importance |
| ⚡ **Interactive** | Streamlit sliders — adjust any feature and re-predict instantly |

---

## 🧪 Dataset

**ClinVar Conflicting Variants** — publicly available from the NIH National Center for Biotechnology Information.

- Download: [Kaggle — ClinVar Conflicting](https://www.kaggle.com/datasets/kevinarvai/clinvar-conflicting)
- Place the file `clinvar_conflicting.csv` in the project root
- If the file is absent, the app generates a realistic synthetic dataset automatically

**Features used:**

| Feature | Meaning |
|---|---|
| `AF_EXAC` | Population allele frequency (rare = more likely harmful) |
| `SIFT_score` | Protein damage predictor (< 0.05 = damaging) |
| `Polyphen2_HVAR_score` | Pathogenicity score (> 0.85 = probably damaging) |
| `CADD_phred` | Combined damage score (> 20 = significant) |
| `LoF` | Loss-of-Function flag |
| `var_type` | SNV / Deletion / Insertion / Duplication |
| `CHROM` | Chromosome number |

---

## 🛠️ Tech Stack

- **Python 3.10**
- **Scikit-learn** — Random Forest classifier
- **SHAP** — Explainable AI / feature attribution
- **Streamlit** — Web application framework
- **Pandas / NumPy** — Data processing
- **Matplotlib** — Visualisations

---

## ⚙️ Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/cancer-gene-mutation-explainer.git
cd cancer-gene-mutation-explainer
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. (Optional) Download the dataset**

Download `clinvar_conflicting.csv` from [Kaggle](https://www.kaggle.com/datasets/kevinarvai/clinvar-conflicting) and place it in the project root.  
*(The app works without it using synthetic data.)*

**4. Run the app**
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## ☁️ Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select this repo → set main file to `app.py`
4. Click **Deploy** — your live link is ready in ~2 minutes!

---

## 📁 Project Structure

```
cancer-gene-mutation-explainer/
│
├── app.py                   # Main Streamlit web application
├── analysis_notebook.ipynb  # Step-by-step analysis & model training
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## 📈 Model Performance

| Metric | Score |
|---|---|
| Accuracy | ~89% (real dataset) |
| Model | Random Forest (100 trees) |
| Explainability | SHAP TreeExplainer |

---

## 💡 What Makes This Different

Most ML projects just show accuracy numbers. This project uses **Explainable AI (SHAP)** — the same technique used by pharma companies and the FDA to understand why models make predictions. This is one of the most in-demand skills in biotech AI in 2026.

---

## ⚠️ Disclaimer

This project is for **educational and portfolio purposes only**. It is not intended for clinical or diagnostic use. Always consult a qualified medical geneticist for real variant interpretation.

---

## 📄 License

MIT License — free to use, modify, and share.

---

## 🙋 Author

**Shivaharish S**  
[LinkedIn](https://linkedin.com) | [GitHub](https://github.com)

*Built as part of a portfolio in AI/ML applied to Biotechnology.*
