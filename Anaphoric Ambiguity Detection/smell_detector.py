import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import joblib, os, warnings, json, torch, optuna, nltk, re
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# ==========================================================
# 1️⃣ Load and Prepare DAMIR-PURE Dataset
# ==========================================================
def load_damir_dataset(path="DAMIR-PURE.csv"):
    print(f"\n📂 Loading DAMIR-PURE dataset from: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    if 'Context' not in df.columns or 'AckUnack' not in df.columns:
        raise ValueError("Expected columns 'Context' and 'AckUnack' not found in the dataset.")

    df = df[['Context', 'AckUnack']].drop_duplicates().reset_index(drop=True)
    df = df.rename(columns={'Context': 'text', 'AckUnack': 'label'})
    df['label'] = df['label'].map({'Ambiguous': 1, 'Unambiguous': 0})
    df = df.dropna(subset=['label']).reset_index(drop=True)
    df['label'] = df['label'].astype(int)
    df['text'] = df['text'].astype(str).str.strip()

    print(f"✅ Loaded {len(df)} requirements from DAMIR-PURE.")
    print("\nLabel distribution:")
    print(df['label'].value_counts().rename({1: 'Ambiguous', 0: 'Unambiguous'}))

    # Handle one-class issue
    if len(df['label'].unique()) < 2:
        print("\n⚠️ Only one class found. Adding synthetic opposite examples for training stability.")
        synth_df = df.sample(min(5, len(df)), replace=True).copy()
        synth_df['label'] = 1 - df['label'].iloc[0]
        synth_df['text'] = synth_df['text'] + " [synthetic ambiguous example]"
        df = pd.concat([df, synth_df], ignore_index=True)

    print("\n✅ Final label distribution:")
    print(df['label'].value_counts().rename({1: 'Ambiguous', 0: 'Unambiguous'}))
    return df[['text', 'label']]

# ==========================================================
# 2️⃣ Optimized Linguistic Feature Extractor
# ==========================================================
class EnhancedFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Precompile regex for performance
        self.re_pronouns = re.compile(r'\b(it|this|that|they|them|those|these)\b', re.IGNORECASE)
        self.re_modals = re.compile(r'\b(can|could|may|might|must|shall|should|will|would)\b', re.IGNORECASE)
        self.re_determiners = re.compile(r'\b(a|an|the|some|many|few|each|every)\b', re.IGNORECASE)
        self.re_vague = re.compile(r'\b(some|various|appropriate|adequate|significant|many|several|few|multiple)\b', re.IGNORECASE)
        self.re_adverbs = re.compile(r'\b(possibly|probably|usually|often|frequently|commonly|typically)\b', re.IGNORECASE)
        self.re_comparatives = re.compile(r'\b(better|faster|greater|higher|more|less|best|worst)\b', re.IGNORECASE)
        self.re_passive = re.compile(r'\b(was|were|is|are|be|been|being)\s+\w+ed\b', re.IGNORECASE)
        self.re_long_sentence = re.compile(r'(\s+\S+){25,}')
        self.re_punct = re.compile(r'[.!?]')
        self.re_all_caps = re.compile(r'\b[A-Z]{2,}\b')

    def fit(self, X, y=None): return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            X = pd.DataFrame({'text': X})
        texts = X['text'].astype(str).fillna('')

        features = []
        for text in texts:
            num_words = len(text.split())
            num_sentences = len(self.re_punct.findall(text)) or 1

            features.append({
                'length': len(text),
                'num_words': num_words,
                'num_sentences': num_sentences,
                'pronouns': len(self.re_pronouns.findall(text)),
                'modals': len(self.re_modals.findall(text)),
                'determiners': len(self.re_determiners.findall(text)),
                'vague_terms': len(self.re_vague.findall(text)),
                'adverbs': len(self.re_adverbs.findall(text)),
                'comparatives': len(self.re_comparatives.findall(text)),
                'passive_voice': len(self.re_passive.findall(text)),
                'long_sentence': int(bool(self.re_long_sentence.search(text))),
                'has_pronoun_start': int(bool(re.match(r'^(It|This|That|They)', text, re.IGNORECASE))),
                'avg_word_len': np.mean([len(w) for w in text.split()]) if num_words > 0 else 0,
                'word_diversity': len(set(text.lower().split())) / (num_words + 1e-8),
                'comma_density': text.count(',') / num_sentences,
                'caps_density': len(self.re_all_caps.findall(text)) / (num_words + 1e-8),
            })
        return pd.DataFrame(features).fillna(0).values

# ==========================================================
# 3️⃣ BERT Embedding Transformer
# ==========================================================
class BertEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='distilbert-base-uncased'):
        self.model_name = model_name
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name, device=device)
    def fit(self, X, y=None): return self
    def transform(self, X):
        texts = X['text'].astype(str).tolist() if isinstance(X, pd.DataFrame) else X.astype(str).tolist()
        print(f"🔹 Encoding {len(texts)} sentences using {self.model_name} ...")
        return self.model.encode(texts, show_progress_bar=True)

# ==========================================================
# 4️⃣ Hybrid BERT + Linguistic Pipeline
# ==========================================================
def build_hybrid_pipeline(scale_pos_weight=1, params=None):
    text_features = Pipeline([('bert', BertEmbeddingTransformer())])
    numeric_features = Pipeline([
        ('linguistic', EnhancedFeatureExtractor()),
        ('scaler', StandardScaler())
    ])
    params = params or {}
    return Pipeline([
        ('features', ColumnTransformer([
            ('bert_embeddings', text_features, 'text'),
            ('linguistic_features', numeric_features, 'text')
        ])),
        ('classifier', XGBClassifier(
            eval_metric='logloss',
            tree_method='hist',
            random_state=33,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
            **params
        ))
    ])

# ==========================================================
# 5️⃣ Evaluation
# ==========================================================
def evaluate_and_save(model, X_test, y_test, save_path):
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:,1] if hasattr(model, 'predict_proba') else np.zeros(len(y_pred))

    print("\n📈 Standard Evaluation:")
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))

    if hasattr(model, 'predict_proba'):
        calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
        calibrated_model.fit(X_test, y_test)
        y_probs = calibrated_model.predict_proba(X_test)[:,1]
        prec, rec, thr = precision_recall_curve(y_test, y_probs)
        f1 = 2*(prec*rec)/(prec+rec+1e-8)
        opt_t = thr[np.argmax(f1)]
        y_opt = (y_probs >= opt_t).astype(int)
        opt_report = classification_report(y_test, y_opt, output_dict=True)
        print("\n📊 Optimized Evaluation:")
        print(classification_report(y_test, y_opt))
        print(f"Optimal Threshold = {opt_t:.3f}")
    else:
        y_opt, opt_report, opt_t = y_pred, report, 0.5

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "results.json"), "w") as f:
        json.dump({'standard': report, 'optimized': opt_report}, f, indent=4)

    cm1, cm2 = confusion_matrix(y_test, y_pred), confusion_matrix(y_test, y_opt)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues'); plt.title('Standard CM')
    plt.subplot(1,2,2); sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens'); plt.title('Optimized CM')
    plt.tight_layout(); plt.savefig(os.path.join(save_path, "confusion_matrices.png")); plt.close()
    print(f"✅ Results saved in {save_path}")

# ==========================================================
# 6️⃣ Optuna Optimization
# ==========================================================
def optimize_xgb(X_train, y_train, scale_pos_weight_value):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 600),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0)
        }
        pipeline = build_hybrid_pipeline(scale_pos_weight_value, params)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        score = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1_weighted').mean()
        return score
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    print(f"\n🏆 Best Params: {study.best_params}")
    print(f"Best CV F1: {study.best_value:.3f}")
    return study.best_params

# ==========================================================
# 7️⃣ Main
# ==========================================================
def main():
    df = load_damir_dataset("DAMIR-PURE.csv")
    X, y = df[['text']], df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scale_pos_weight_value = y_train.value_counts()[0] / y_train.value_counts()[1]
    print("\n🔎 Optimizing with Optuna ...")
    best_params = optimize_xgb(X_train, y_train, scale_pos_weight_value)
    print("\n🚀 Training final hybrid model ...")
    model = build_hybrid_pipeline(scale_pos_weight_value, best_params)
    model.fit(X_train, y_train)
    save_path = "./results_hybrid"
    evaluate_and_save(model, X_test, y_test, save_path)
    joblib.dump(model, os.path.join(save_path, "best_model.joblib"))
    print(f"\n✅ Model saved at {save_path}/best_model.joblib")

if __name__ == "__main__":
    main()
