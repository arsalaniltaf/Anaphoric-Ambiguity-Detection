# Anaphoric-Ambiguity-Detection

Detection of Anaphoric ambiguity in natural language software requirements

Dataset Source

The dataset used in this project is the DAMIR-PURE dataset, originally released as part of the study by Ezzini et al. (2022). The dataset contains manually annotated software requirements focusing on anaphoric ambiguity, and it is widely used in research on Natural Language Requirements Engineering.
I obtained the dataset from the authors’ public release accompanying their ICSE 2022 paper:
Citation:
Ezzini, S., Abualhaija, S., Arora, C., & Sabetzadeh, M. (2022). Automated Handling of Anaphoric Ambiguity in Requirements: A Multi-solution Study. Proceedings of the 44th International Conference on Software Engineering (ICSE), 187–199.

Hybrid Ambiguity Detection Model:

This repository provides a hybrid NLP + Machine Learning pipeline for detecting ambiguity in natural language software requirements.
The approach integrates BERT-based semantic embeddings with linguistic handcrafted features, optimized using XGBoost and Optuna.
📌 1. Overview
The system performs:
Dataset loading & cleaning
Linguistic feature extraction
BERT embedding generation
Training a hybrid feature pipeline
Hyperparameter tuning via Optuna
Performance evaluation (standard + optimized threshold)
Saving results & trained model
All outputs are automatically stored in a results folder.
📁 2. Project Structure

project_folder/
│

├── smell_detector.py

├── DAMIR-PURE.csv

└── results_hybrid/        # created after running the script

 3. Methodology
A. Linguistic Features
The script extracts handcrafted features such as:
Pronouns
Modal verbs
Vague expressions
Passive voice
Word diversity
Average word length
All-caps density
Comma density
Long sentence indicator
Pronoun-start indicator
B. BERT Embeddings
Using:
SentenceTransformer("distilbert-base-uncased")
This provides dense semantic representations for each requirement sentence.
C. Hybrid Pipeline
Merges BERT + linguistic features via ColumnTransformer
Trains XGBClassifier
Handles class imbalance
Runs 3-fold Optuna tuning for optimal parameters
D. Evaluation
Outputs include:
Standard classification report
Optimized threshold report
Confusion matrices
JSON metrics
Saved trained model
💻 4. Installation
Install required dependencies:
pip install pandas numpy matplotlib seaborn scikit-learn xgboost optuna sentence-transformers joblib nltk
▶️ 5. How to Run
Mac (macOS)
cd /path/to/project_folder
python3 smell_detector.py
Windows
cd C:\path\to\project_folder
python smell_detector.py
The script will automatically create:
results_hybrid/
containing all evaluation files and the model.
📊 6. Output Files
File	Description
results.json	Standard + optimized performance metrics
confusion_matrices.png	Standard vs optimized confusion matrices
best_model.joblib	Final trained model
Console Output	Optuna logs and training details
⚙️ 7. Notes
Make sure DAMIR-PURE.csv is in the same directory as the script.
GPU support:
NVIDIA GPU → CUDA
Apple Silicon → MPS
Optuna tuning may take a few minutes depending on system performance.
