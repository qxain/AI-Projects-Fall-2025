Hallucination Detector Project

Author: Quraitul Ain
Date: Fall 2025
Course: Artificial Intelligence Final Project

Objective
Detect unsupported (hallucinated) claims using the FEVER Claim Extraction dataset and demonstrate lightweight mitigation strategies that reduce hallucination errors without altering model architecture.
This project aligns with the research theme “LLM Hallucinations: From Guessing to Knowing.”

Project Summary
The project builds a simple, interpretable text-classification pipeline that:
Identifies whether a textual claim is supported (factual) or unsupported (hallucinated).
Uses TF-IDF vectorization with a Logistic Regression model.
Applies three lightweight mitigation strategies:
Threshold Optimization – tune decision cutoff for best F1-score.
Selective Abstention – abstain on low-confidence predictions.
Simulated RAG Fix – estimate improvement if retrieval grounding corrected abstained claims.
Together, these techniques help the system “know when it doesn’t know,” improving factual reliability.

Dataset
Dataset: KnutJaegersberg/FEVER_claim_extraction
Source: Hugging Face
Field	Description
claim	Short statement or factual assertion
y	Binary label → 0 = supported, 1 = unsupported/hallucinated

Workflow Overview
Step	Stage	Description
1	Setup	Import dependencies and set random seed.
2	Load Dataset	Load FEVER dataset and extract claim + y columns.
3	Explore Data	View class distribution and sample rows.
4	Preprocess Text	Convert text into numerical TF-IDF vectors (max 5 k features).
5	Train Model	Fit Logistic Regression classifier on the training data.
6	Evaluate	Compute metrics: accuracy, F1, ROC-AUC, PR-curve.
7	Optimize Threshold	Tune decision threshold to maximize F1 for class 1.
8	Selective Abstention	Filter out low-confidence predictions (confidence < τ).
9	Cumulative Improvement	Plot Baseline → Opt → Abstain → RAG Fix error reduction.
10	Error Analysis	Display correct, incorrect, and abstained claims.
11	Conclusion	Summarize improvements and tie back to project theme.

Results Summary
Baseline Accuracy: (see notebook output)
Optimized F1 (Hallucination): ↑ after threshold tuning
Abstention Coverage: ~80–90 % (depending on confidence cutoff)
Overall Effect: Reduced hallucination misclassifications with minimal complexity.


Key Visuals
Confusion Matrix – shows where model confuses supported vs hallucinated claims.
ROC Curve – true/false-positive trade-off (higher AUC = better).
Precision–Recall Curve – better insight under class imbalance.
Improvement Graph – visualizes cumulative reduction in error across mitigation stages.

How to Run the Project
git clone https://github.com/aalomari-ctrl/AI-Projects-Fall-2025.git
cd AI-Projects-Fall-2025/Hallucination_Detector_Project

Create and Activate a Virtual Environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows

Install Dependencies
pip install -r requirements.txt
pip install numpy pandas scikit-learn matplotlib seaborn datasets

Run in jupyter notebook or Colab

References
OpenAI (2024): Why Language Models Hallucinate
FEVER Dataset on Hugging Face
scikit-learn Documentation

# Core libraries
numpy==1.26.4
pandas==2.2.2

# Machine learning
scikit-learn==1.5.2

# Visualization
matplotlib==3.9.2
seaborn==0.13.2

# Dataset handling
datasets==3.0.1
huggingface-hub==0.25.2

# Notebook environment
jupyter==1.1.1
ipykernel==6.29.5

# Optional utilities
tqdm==4.66.5
requests==2.32.3







