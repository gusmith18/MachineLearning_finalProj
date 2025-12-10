import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve


# Config
CSV = 'second_features.csv'
UNNEEDED = ['name', 'type', 'subtypes', 'supertypes', 'colors',
            'color_identity', 'rarity', 'text', 'flavor', 'image_url',
            'mana_cost', 'artist']
Y_COLUMNS = ["is_white", "is_blue", "is_black", "is_red", "is_green"]
OUT_DIR = 'plots'
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    # Read data (same filtering you used)
    df = pd.read_csv(CSV, low_memory=False)
    df = df.dropna(subset=['red_pct', 'green_pct', 'blue_pct', 'yellow_pct', 'black_pct', 'white_pct'])
    df = df.set_index('multiverse_id')
    features = df.drop(columns=UNNEEDED)
    y = features[Y_COLUMNS]
    features = features.drop(Y_COLUMNS, axis=1)

    X_train, X_test, y_train_df, y_test_df = train_test_split(features, y, test_size=0.2, random_state=42)

    # Shared preprocessing pipeline
    preproc = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    X_train_p = preproc.fit_transform(X_train)
    X_test_p = preproc.transform(X_test)

    # Architectures to compare (list of tuples)
    ARCHITECTURES = [
        (100, 50),
        (200, 100, 50),
        (300, 200, 100, 50)
    ]

    metrics_rows = []

    # Colors for plotting (will cycle if more architectures)
    plot_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Loop over each binary target and each architecture
    for i, col in enumerate(Y_COLUMNS):
        y_train = y_train_df[col].astype(int)
        y_test = y_test_df[col].astype(int)

        # store per-arch metrics and ROC curves for this color
        per_arch_results = []

        for a_idx, arch in enumerate(ARCHITECTURES):
            mlp = MLPClassifier(hidden_layer_sizes=arch, max_iter=300, random_state=42, early_stopping=True, validation_fraction=0.1)
            mlp.fit(X_train_p, y_train)
            y_pred = mlp.predict(X_test_p)
            y_proba = mlp.predict_proba(X_test_p)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else float('nan')

            per_arch_results.append({'arch': arch, 'acc': acc, 'f1': f1, 'auc': auc, 'y_proba': y_proba, 'model': mlp})

            metrics_rows.append({'color': col, 'arch': str(arch), 'acc': acc, 'f1': f1, 'auc': auc})

        # Plot ROC curves for this color with all architectures
        plt.figure(figsize=(6,5))
        for a_idx, r in enumerate(per_arch_results):
            if not np.isnan(r['auc']):
                fpr, tpr, _ = roc_curve(y_test, r['y_proba'])
                plt.plot(fpr, tpr, label=f'{r["arch"]} (AUC={r["auc"]:.3f})', color=plot_colors[a_idx % len(plot_colors)])
        plt.plot([0,1],[0,1],'k--', alpha=0.4)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC - {col}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'roc_{col}.png'))
        plt.close()

    # Save metrics
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv('feature_model_comparison.csv', index=False)

    # Bar plots for Accuracy and F1 across architectures
    # Pivot so architectures become columns
    acc_pivot = metrics_df.pivot(index='color', columns='arch', values='acc')
    f1_pivot = metrics_df.pivot(index='color', columns='arch', values='f1')

    colors = plot_colors
    n_arch = acc_pivot.shape[1]
    x = np.arange(len(acc_pivot.index))
    total_width = 0.8
    width = total_width / max(1, n_arch)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Accuracy grouped bar chart
    for i, col_arch in enumerate(acc_pivot.columns):
        ax[0].bar(x - total_width/2 + i*width + width/2, acc_pivot[col_arch].values, width, label=str(col_arch), color=colors[i % len(colors)])
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(acc_pivot.index, rotation=45)
    ax[0].set_title('Accuracy by Color and Architecture')
    ax[0].legend(title='arch')

    # F1 grouped bar chart
    for i, col_arch in enumerate(f1_pivot.columns):
        ax[1].bar(x - total_width/2 + i*width + width/2, f1_pivot[col_arch].values, width, label=str(col_arch), color=colors[i % len(colors)])
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(f1_pivot.index, rotation=45)
    ax[1].set_title('F1 Score by Color and Architecture')
    ax[1].legend(title='arch')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'accuracy_f1.png'))
    plt.close()

    # MLP loss curve (if available)
    if hasattr(mlp, 'loss_curve_'):
        plt.figure(figsize=(6,4))
        plt.plot(mlp.loss_curve_)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('MLP Loss Curve (last trained MLP)')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'mlp_loss.png'))
        plt.close()

    print('Comparison complete. Metrics saved to feature_model_comparison.csv and plots saved in', OUT_DIR)


if __name__ == '__main__':
    main()
