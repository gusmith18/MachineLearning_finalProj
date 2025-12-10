"""
Visualization of RandomForest card color prediction results.

Generates plots:
1. Confusion matrix heatmap
2. Feature importance bar chart (top 20)
3. Class distribution
4. Model accuracy comparison (train vs test)
5. Prediction probability distribution
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

# Files
DATA_FILE = 'second_features.csv'
MODEL_FILE = 'card_color_rf_model.joblib'
OUTPUT_DIR = 'rf_plots'

# Color mapping
COLOR_MAP = {
    "['W']": 'W',
    "['U']": 'U',
    "['B']": 'B',
    "['R']": 'R',
    "['G']": 'G',
}

MAGIC_COLORS = {
    'W': '#F0E68C',  # White/Yellow
    'U': '#4169E1',  # Blue
    'B': '#2F4F4F',  # Black
    'R': '#DC143C',  # Red
    'G': '#228B22',  # Green
}


def load_data_and_model():
    """Load data and trained model."""
    print("Loading data and model...")
    df = pd.read_csv(DATA_FILE)
    df['color_label'] = df['colors'].astype(str).map(COLOR_MAP)
    df_filtered = df[df['color_label'].notna()].copy()

    loaded = joblib.load(MODEL_FILE)
    # Support both (clf, feature_cols) and (clf, feature_cols, color_feats)
    if isinstance(loaded, tuple):
        if len(loaded) >= 2:
            clf = loaded[0]
            feature_cols = loaded[1]
            color_feats = loaded[2] if len(loaded) > 2 else None
        else:
            clf = loaded[0]
            feature_cols = None
            color_feats = None
    else:
        clf = loaded
        feature_cols = None
        color_feats = None

    return df_filtered, clf, feature_cols, color_feats


def prepare_predictions(df, clf, feature_cols, color_feats=None):
    """Prepare data for visualization."""
    exclude_cols = {
        'name', 'multiverse_id', 'mana_cost', 'colors', 'color_identity', 'type',
        'supertypes', 'subtypes', 'rarity', 'text', 'flavor', 'artist', 
        'power', 'toughness', 'image_url', 'color_label'
    }
    
    feature_cols_filtered = [c for c in feature_cols if c not in exclude_cols and c in df.columns]
    X = df[feature_cols_filtered].fillna(0).astype(np.float32)
    y = df['color_label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Predictions
    y_pred_train_raw = clf.predict(X_train)
    y_pred_test_raw = clf.predict(X_test)

    # Attempt to get prediction probabilities for train and test
    try:
        y_pred_proba_train_raw = clf.predict_proba(X_train)
        y_pred_proba_test_raw = clf.predict_proba(X_test)
    except Exception:
        y_pred_proba_train_raw = None
        y_pred_proba_test_raw = None

    # If classifier is multi-output (preds are 2D arrays), convert to single-label
    # by selecting the color with highest positive probability per sample.
    if isinstance(y_pred_test_raw, np.ndarray) and y_pred_test_raw.ndim == 2:
        # Build positive-probability matrix (n_samples, n_colors)
        if isinstance(y_pred_proba_test_raw, list):
            pos_probs_test = np.vstack([p[:, 1] if p.shape[1] > 1 else p[:, 0] for p in y_pred_proba_test_raw]).T
        else:
            # fallback: use raw predictions (0/1) as scores
            pos_probs_test = y_pred_test_raw.astype(float)

        if isinstance(y_pred_proba_train_raw, list):
            pos_probs_train = np.vstack([p[:, 1] if p.shape[1] > 1 else p[:, 0] for p in y_pred_proba_train_raw]).T
        elif y_pred_proba_train_raw is None:
            pos_probs_train = y_pred_train_raw.astype(float)
        else:
            pos_probs_train = y_pred_proba_train_raw

        # Map color feature names to single-letter color codes
        feat_to_letter = {
            'is_white': 'W', 'is_blue': 'U', 'is_black': 'B', 'is_red': 'R', 'is_green': 'G'
        }
        if color_feats is None:
            # default ordering (if color_feats missing)
            color_order = ['is_blue', 'is_red', 'is_black', 'is_green', 'is_white']
        else:
            color_order = list(color_feats)

        color_letters = [feat_to_letter.get(f, f) for f in color_order]

        idx_train = np.argmax(pos_probs_train, axis=1)
        idx_test = np.argmax(pos_probs_test, axis=1)

        y_pred_train = [color_letters[i] for i in idx_train]
        y_pred_test = [color_letters[i] for i in idx_test]

        # For plotting confidence, use the positive-prob matrix for test
        y_pred_proba = pos_probs_test
        clf_classes = color_letters
    else:
        # Single-label classifier
        y_pred_train = y_pred_train_raw
        y_pred_test = y_pred_test_raw
        # predict_proba returns (n_samples, n_classes)
        y_pred_proba = y_pred_proba_test_raw
        clf_classes = clf.classes_

    return X_train, X_test, y_train, y_test, y_pred_train, y_pred_test, y_pred_proba, clf_classes


def plot_confusion_matrix(y_test, y_pred, clf_classes):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_test, y_pred, labels=clf_classes)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=clf_classes, yticklabels=clf_classes,
                cbar_kws={'label': 'Count'}, ax=ax)
    ax.set_xlabel('Predicted Color', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual Color', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix - Card Color Prediction', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/01_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 01_confusion_matrix.png")
    plt.close()


def plot_feature_importance(clf, feature_cols, top_n=20):
    """Plot top N feature importances."""
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors_bar = plt.cm.viridis(np.linspace(0, 1, top_n))
    bars = ax.barh(range(top_n), importances[indices], color=colors_bar)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_cols[i] for i in indices])
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/02_feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 02_feature_importance.png")
    plt.close()


def plot_accuracy_comparison(y_train, y_pred_train, y_test, y_pred_test):
    """Plot train vs test accuracy."""
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy bars
    accs = [train_acc, test_acc]
    labels = ['Train', 'Test']
    colors = ['#2ecc71', '#3498db']
    bars = ax1.bar(labels, accs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_ylim([0.99, 1.0])
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    # Add value labels
    for bar, acc in zip(bars, accs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height,
                f'{acc:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Error rate
    errors = [1 - train_acc, 1 - test_acc]
    bars2 = ax2.bar(labels, errors, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Error Rate', fontsize=12, fontweight='bold')
    ax2.set_title('Model Error Rate', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    
    # Add value labels
    for bar, err in zip(bars2, errors):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height * 1.5,
                f'{err:.6f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/03_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 03_accuracy_comparison.png")
    plt.close()


def plot_class_distribution(y):
    """Plot class distribution."""
    counts = y.value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_list = [MAGIC_COLORS[c] for c in counts.index]
    bars = ax.bar(counts.index, counts.values, color=colors_list, alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_xlabel('Card Color', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Card Color Distribution', fontsize=14, fontweight='bold')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
                f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/04_class_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 04_class_distribution.png")
    plt.close()


def plot_prediction_confidence(y_test, y_pred_proba, clf_classes):
    """Plot prediction confidence distribution."""
    max_proba = np.max(y_pred_proba, axis=1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of max probabilities
    ax1.hist(max_proba, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(x=np.mean(max_proba), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(max_proba):.4f}')
    ax1.axvline(x=np.median(max_proba), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(max_proba):.4f}')
    ax1.set_xlabel('Prediction Confidence (Max Probability)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Confidence by color
    confidence_by_color = {}
    for color in clf_classes:
        mask = y_test == color
        if mask.sum() > 0:
            confidence_by_color[color] = max_proba[mask]
    
    bp = ax2.boxplot([confidence_by_color[c] for c in clf_classes],
                      labels=clf_classes, patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], [MAGIC_COLORS[c] for c in clf_classes]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_xlabel('Card Color', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Prediction Confidence', fontsize=12, fontweight='bold')
    ax2.set_title('Confidence by Card Color', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/05_prediction_confidence.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 05_prediction_confidence.png")
    plt.close()


def plot_summary_dashboard(y_train, y_pred_train, y_test, y_pred_test, y, clf_classes):
    """Create a summary dashboard."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('RandomForest Card Color Prediction - Summary Dashboard', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 1. Accuracy
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(['Train', 'Test'], [train_acc, test_acc], color=['#2ecc71', '#3498db'], alpha=0.7, edgecolor='black')
    ax1.set_ylim([0.99, 1.0])
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy')
    for i, (label, acc) in enumerate(zip(['Train', 'Test'], [train_acc, test_acc])):
        ax1.text(i, acc, f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Class distribution
    ax2 = fig.add_subplot(gs[0, 1])
    counts = y.value_counts().sort_index()
    colors_list = [MAGIC_COLORS[c] for c in counts.index]
    ax2.bar(counts.index, counts.values, color=colors_list, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Count')
    ax2.set_title('Class Distribution')
    
    # 3. Train/Test split
    ax3 = fig.add_subplot(gs[0, 2])
    sizes = [len(y_train), len(y_test)]
    ax3.pie(sizes, labels=[f'Train\n({sizes[0]})', f'Test\n({sizes[1]})'], autopct='%1.1f%%',
            colors=['#2ecc71', '#3498db'], startangle=90)
    ax3.set_title('Train/Test Split')
    
    # 4. Confusion matrix (mini)
    ax4 = fig.add_subplot(gs[1, :2])
    cm = confusion_matrix(y_test, y_pred_test, labels=clf_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=clf_classes, yticklabels=clf_classes,
                cbar_kws={'label': 'Count'}, ax=ax4, square=True)
    ax4.set_title('Confusion Matrix (Test Set)')
    ax4.set_xlabel('Predicted')
    ax4.set_ylabel('Actual')
    
    # 5. Top features
    ax5 = fig.add_subplot(gs[1, 2])
    # This will be added after loading model
    ax5.text(0.5, 0.5, 'Feature importance\n(see separate plot)', 
             ha='center', va='center', transform=ax5.transAxes)
    ax5.set_title('Top Features')
    ax5.axis('off')
    
    # 6. Per-class metrics
    ax6 = fig.add_subplot(gs[2, :])
    from sklearn.metrics import precision_score, recall_score, f1_score
    precisions = [precision_score(y_test, y_pred_test, labels=[c], average='weighted', zero_division=0) 
                  for c in clf_classes]
    recalls = [recall_score(y_test, y_pred_test, labels=[c], average='weighted', zero_division=0) 
               for c in clf_classes]
    f1s = [f1_score(y_test, y_pred_test, labels=[c], average='weighted', zero_division=0) 
           for c in clf_classes]
    
    x = np.arange(len(clf_classes))
    width = 0.25
    ax6.bar(x - width, precisions, width, label='Precision', alpha=0.8)
    ax6.bar(x, recalls, width, label='Recall', alpha=0.8)
    ax6.bar(x + width, f1s, width, label='F1-Score', alpha=0.8)
    ax6.set_ylabel('Score')
    ax6.set_title('Per-Color Metrics (Test Set)')
    ax6.set_xticks(x)
    ax6.set_xticklabels(clf_classes)
    ax6.legend()
    ax6.set_ylim([0.99, 1.01])
    ax6.grid(alpha=0.3, axis='y')
    
    plt.savefig(f'{OUTPUT_DIR}/00_summary_dashboard.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 00_summary_dashboard.png")
    plt.close()


def main():
    """Generate all plots."""
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading data and model...")
    df, clf, feature_cols, color_feats = load_data_and_model()
    
    print("Preparing predictions...")
    X_train, X_test, y_train, y_test, y_pred_train, y_pred_test, y_pred_proba, clf_classes = prepare_predictions(df, clf, feature_cols, color_feats)
    
    print("\nGenerating plots...\n")
    plot_summary_dashboard(y_train, y_pred_train, y_test, y_pred_test, df['color_label'], clf_classes)
    plot_confusion_matrix(y_test, y_pred_test, clf_classes)
    plot_feature_importance(clf, feature_cols, top_n=20)
    plot_accuracy_comparison(y_train, y_pred_train, y_test, y_pred_test)
    plot_class_distribution(df['color_label'])
    plot_prediction_confidence(y_test, y_pred_proba, clf_classes)
    
    print(f"\n✓ All plots saved to {OUTPUT_DIR}/ directory")
    print("\nFiles generated:")
    print("  - 00_summary_dashboard.png")
    print("  - 01_confusion_matrix.png")
    print("  - 02_feature_importance.png")
    print("  - 03_accuracy_comparison.png")
    print("  - 04_class_distribution.png")
    print("  - 05_prediction_confidence.png")


if __name__ == '__main__':
    main()
