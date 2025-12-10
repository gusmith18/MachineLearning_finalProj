"""
RandomForest classifier to predict Magic card color from card features.

This script:
1. Loads second_features.csv
2. Extracts color from 'colors' column (W, U, B, R, G)
3. Uses ability/type/rarity features to train a RandomForest
4. Evaluates on test set and saves the trained model
5. Provides predictions and feature importance
"""

import pandas as pd
import numpy as np
import time
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# File paths
DATA_FILE = 'second_features.csv'
MODEL_OUT = 'card_color_rf_model.joblib'
REPORT_OUT = 'card_color_rf_report.txt'

# Color mapping
COLOR_MAP = {
    "['W']": 'W',
    "['U']": 'U',
    "['B']": 'B',
    "['R']": 'R',
    "['G']": 'G',
}


def load_and_prepare_data(csv_file):
    """Load CSV and prepare data for training."""
    print(f"Loading {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns\n")
    
    # Extract single-color cards only (for simpler classification)
    print("Filtering to single-color cards...")
    df['color_label'] = df['colors'].astype(str).map(COLOR_MAP)
    df_filtered = df[df['color_label'].notna()].copy()
    print(f"Filtered to {len(df_filtered)} single-color cards\n")
    
    # Identify feature columns (everything except a few metadata columns)
    # and exclude the five color indicator features from X. The five color
    # indicator features will be the multi-output target Y.
    exclude_cols = {
        'name', 'multiverse_id', 'mana_cost', 'colors', 'color_identity', 'type',
        'supertypes', 'subtypes', 'rarity', 'text', 'flavor', 'artist',
        'power', 'toughness', 'image_url', 'color_label'
    }

    # Define the five color indicator features (these will be the target)
    color_feats = ['is_blue', 'is_red', 'is_black', 'is_green', 'is_white']

    # X = all columns except excluded metadata and the five color features
    feature_cols = [c for c in df_filtered.columns if c not in exclude_cols and c not in color_feats]
    print(f"Using {len(feature_cols)} features for training")
    print(f"Features (sample): {', '.join(feature_cols[:20])}...")

    X = df_filtered[feature_cols].fillna(0).astype(np.float32)

    # y is the multi-output DataFrame of the five color indicator columns
    # Ensure binary integer values (0/1)
    y = df_filtered[color_feats].fillna(0).astype(int)

    print(f"\nFeature matrix shape: {X.shape}")
    print("Target distribution (positive counts per color):")
    for col in color_feats:
        print(f"  {col}: {y[col].sum()} / {len(y)}")

    return X, y, feature_cols


def train_model(X, y, feature_cols):
    """Train RandomForest classifier."""
    print("="*60)
    print("Training RandomForest Classifier")
    print("="*60 + "\n")
    
    start = time.time()
    
    # Split data (stratify not supported for multioutput targets)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples\n")
    
    # Train model
    print("Fitting RandomForest with 300 trees, max_depth=15...")
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced'
    )
    clf.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"Training completed in {elapsed:.1f}s\n")
    
    # Evaluate
    print("="*60)
    print("Evaluation Results")
    print("="*60 + "\n")
    
    # Multi-output predictions
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    # Overall subset accuracy (all labels must match)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    print(f"Training (subset) Accuracy: {train_acc:.4f}")
    print(f"Test (subset) Accuracy: {test_acc:.4f}\n")

    # Per-label reports
    print("Per-Label Classification Reports (Test Set):")
    for i, col in enumerate(y.columns):
        print(f"\n--- {col} ---")
        print(classification_report(y_test.iloc[:, i], y_pred_test[:, i]))

    # Per-label confusion matrices
    print("\nPer-Label Confusion Matrices (Test Set):")
    for i, col in enumerate(y.columns):
        print(f"\n{col}:")
        print(confusion_matrix(y_test.iloc[:, i], y_pred_test[:, i]))

    # Cross-validation per label (5-fold)
    print("\n" + "="*60)
    print("Cross-Validation (5-fold) per label")
    print("="*60)
    cv_results = {}
    for col in y.columns:
        scores = cross_val_score(clf, X_train, y_train[col], cv=5, n_jobs=-1)
        cv_results[col] = scores
        print(f"{col}: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    # Feature importance
    print("="*60)
    print("Top 20 Most Important Features")
    print("="*60 + "\n")
    
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:20]

    for rank, idx in enumerate(indices, 1):
        feat_name = feature_cols[idx]
        importance = importances[idx]
        print(f"{rank:2d}. {feat_name:30s}: {importance:.4f}")
    
    return clf, (X_train, X_test, y_train, y_test), (train_acc, test_acc)


def save_results(clf, feature_cols, train_acc, test_acc, color_feats=None):
    """Save model and report."""
    # Save model
    # Save (clf, feature_cols, color_feature_names) for later loading
    to_save = (clf, feature_cols, list(color_feats) if color_feats is not None else None)
    joblib.dump(to_save, MODEL_OUT)
    print(f"\n✓ Model saved to {MODEL_OUT}")
    
    # Save text report
    with open(REPORT_OUT, 'w') as f:
        f.write("="*60 + "\n")
        f.write("RandomForest Card Color Classifier (multi-output)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Features used: {len(feature_cols)}\n")
        f.write(f"Training (subset) Accuracy: {train_acc:.4f}\n")
        f.write(f"Test (subset) Accuracy: {test_acc:.4f}\n")
        f.write("\nFeature list:\n")
        for i, feat in enumerate(feature_cols, 1):
            f.write(f"  {i}. {feat}\n")

        if color_feats is not None:
            f.write("\nTarget color columns:\n")
            for i, cf in enumerate(color_feats, 1):
                f.write(f"  {i}. {cf}\n")

        f.write("\nNote: This model predicts five binary color indicator columns as multi-output targets.\n")
    
    print(f"✓ Report saved to {REPORT_OUT}")


def predict_color(model_file, card_features):
    """
    Predict card color from feature vector.
    
    Args:
        model_file: path to saved model (joblib file)
        card_features: dict or array of feature values
    
    Returns:
        predicted color (W, U, B, R, G) and probabilities
    """
    loaded = joblib.load(model_file)
    # Support either (clf, feature_cols) or (clf, feature_cols, color_feats)
    if isinstance(loaded, tuple) and len(loaded) >= 2:
        clf, feature_cols = loaded[0], loaded[1]
        color_feats = loaded[2] if len(loaded) > 2 else None
    else:
        clf = loaded
        feature_cols = None
        color_feats = None

    # Convert dict to array if needed
    if isinstance(card_features, dict):
        if feature_cols is None:
            raise ValueError('Model does not include feature column list; pass features as ordered array')
        arr = np.array([card_features.get(col, 0) for col in feature_cols], dtype=np.float32)
    else:
        arr = np.array(card_features, dtype=np.float32).reshape(1, -1)

    if len(arr.shape) == 1:
        arr = arr.reshape(1, -1)

    pred = clf.predict(arr)[0]

    # predict_proba for multi-output returns a list (one per output)
    probs_list = clf.predict_proba(arr)
    probs_map = {}
    # If model was single-output, handle gracefully
    if not isinstance(probs_list, list):
        # single-output classifier
        probs_map = dict(zip(clf.classes_, probs_list[0]))
    else:
        # multi-output: for each output, take probability of positive class (1)
        for i, out_probs in enumerate(probs_list):
            # out_probs shape: (1, n_classes)
            classes = clf.classes_[i] if isinstance(clf.classes_, list) else clf.classes_
            # find index of class '1' (positive)
            if list(classes) == [0, 1] or list(classes) == ['0', '1']:
                p_pos = out_probs[0][list(classes).index(1) if 1 in classes else 1]
            else:
                # fallback: if second column is positive
                p_pos = out_probs[0][-1]

            key = color_feats[i] if color_feats is not None else f'label_{i}'
            probs_map[key] = float(p_pos)

    # Build pred_map (0/1 per color)
    pred_map = {}
    for i, val in enumerate(pred):
        key = color_feats[i] if color_feats is not None else f'label_{i}'
        pred_map[key] = int(val)

    return pred_map, probs_map


def main():
    """Main execution."""
    print("\n" + "="*60)
    print(f"START TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")
    
    # Load data
    X, y, feature_cols = load_and_prepare_data(DATA_FILE)
    
    # Train
    clf, (X_train, X_test, y_train, y_test), (train_acc, test_acc) = train_model(X, y, feature_cols)
    
    # Save (include color target names)
    save_results(clf, feature_cols, train_acc, test_acc, y.columns)
    
    print("\n" + "="*60)
    print(f"END TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")
    
    # Example: predict (multi-output)
    print("Example prediction:")
    sample = X_test.iloc[0]
    pred_map, probs_map = predict_color(MODEL_OUT, sample)
    print(f"  Predicted (per-color): {pred_map}")
    print(f"  Probabilities (positive class per-color): {probs_map}")
    print(f"  Actual targets: {y_test.iloc[0].to_dict()}")


if __name__ == '__main__':
    main()
