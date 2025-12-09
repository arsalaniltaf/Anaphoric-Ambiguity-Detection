Hybrid Ambiguity Detection Model (DAMIR-PURE Dataset)
Author: Muhammad Arsalan
File: smell_detector.py

------------------------------------------------------------
1. OVERVIEW
------------------------------------------------------------
This project implements a hybrid NLP + Machine Learning pipeline
for detecting ambiguity in natural language software requirements
using the DAMIR-PURE dataset.

The system combines:
• BERT sentence embeddings
• Linguistic handcrafted features
• XGBoost classifier
• Optuna hyperparameter tuning

The script loads the dataset, extracts features, trains the model,
evaluates performance, and saves results in a separate folder.

------------------------------------------------------------
2. WHAT THE SCRIPT DOES
------------------------------------------------------------

A) Data Loading
• Loads DAMIR-PURE.csv
• Cleans dataset and converts labels to 0/1
• Removes duplicates and missing values
• Handles one-class issue using synthetic samples

B) Feature Extraction
1. Linguistic Features:
   - Pronouns, modals, determiners
   - Vague terms & passive voice
   - Word diversity, caps density, comma density
   - Long sentence detection

2. BERT Embeddings:
   - Uses SentenceTransformer (distilbert-base-uncased)
   - Generates semantic embeddings for sentences

C) Model Pipeline
• Uses ColumnTransformer to merge BERT + linguistic features
• Trains XGBoost classifier with class imbalance handling

D) Optimization
• Optuna optimizes model hyperparameters using 3-fold CV

E) Evaluation
• Standard and threshold-optimized classification reports
• Confusion matrices (saved as an image)
• Results saved as JSON + PNG

F) Saved Outputs:
• results.json
• confusion_matrices.png
• best_model.joblib

------------------------------------------------------------
3. REQUIREMENTS
------------------------------------------------------------
Python 3.8 – 3.12

Install required libraries:
pip install pandas numpy matplotlib seaborn scikit-learn xgboost optuna sentence-transformers joblib nltk

------------------------------------------------------------
4. FILE STRUCTURE
------------------------------------------------------------
project_folder/
│
├── smell_detector.py
├── DAMIR-PURE.csv
└── results_hybrid/   (auto-created)

------------------------------------------------------------
5. HOW TO RUN THE SCRIPT
------------------------------------------------------------

MACOS:
1. open Terminal
2. go to the folder:
   cd /path/to/project_folder
3. run:
   python3 smell_detector.py
4. results will be created inside results_hybrid/

WINDOWS:
1. open Command Prompt / PowerShell
2. go to the folder:
   cd C:\path\to\project_folder
3. run:
   python smell_detector.py
4. results will be created inside results_hybrid/

------------------------------------------------------------
6. OUTPUT FILES
------------------------------------------------------------
results.json                → evaluation metrics
confusion_matrices.png      → standard + optimized confusion matrices
best_model.joblib           → final trained model

------------------------------------------------------------
7. NOTES
------------------------------------------------------------
• DAMIR-PURE.csv must exist in the same directory.
• GPU acceleration supported (CUDA / Apple MPS).
• Optuna optimization may take time depending on hardware.

------------------------------------------------------------
8. CONTACT
------------------------------------------------------------
For questions or improvements, contact the author.
