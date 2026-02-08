import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_class_distribution(classes_path):
    df = pd.read_csv(classes_path)
    
    # Standardize classes: 1 (Illicit), 2 (Licit), 3 (Unknown)
    df['class'] = df['class'].replace({'unknown': '3'}).astype(int)
    
    stats = df['class'].value_counts().sort_index()
    total = len(df)
    
    illicit_pct = (stats[1] / total) * 100
    ratio = stats[2] / stats[1]

    print(f"\n{'='*40}\nFORENSIC AUDIT: CLASS DISTRIBUTION\n{'='*40}")
    print(f"Illicit (Class 1): {stats[1]:>10,} ({illicit_pct:.2f}%)")
    print(f"Licit   (Class 2): {stats[2]:>10,}")
    print(f"Unknown (Class 3): {stats[3]:>10,}")
    print(f"{'-'*40}\nTotal Nodes: {total:,}\n")
    
    print(f"ANALYST NOTE: Licit-to-Illicit Ratio is {ratio:.1f}:1.")
    print("Action Required: Implement Weighted Loss or Graph-based sampling to address imbalance.\n")

    return df[df['class'].isin([1, 2])]

def plot_distribution(df, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 5))
    sns.countplot(x='class', data=df, palette=['#d9534f', '#5cb85c']) # Professional red/green hex
    plt.title('Bitcoin Transaction Class Distribution')
    plt.xlabel('Forensic Classification (1: Illicit, 2: Licit)')
    plt.ylabel('Node Count')
    plt.xticks([0, 1], ['Illicit', 'Licit'])
    plt.savefig(save_path)

if __name__ == "__main__":
    # Point to the root data folder where the class file actually lives
    CLASSES_PATH = 'data/elliptic_txs_classes.csv'
    SAVE_PATH = 'docs/class_imbalance.png'
    
    import os
    if os.path.exists(CLASSES_PATH):
        labeled_data = analyze_class_distribution(CLASSES_PATH)
        plot_distribution(labeled_data, SAVE_PATH)
    else:
        print(f"❌ ERROR: Still cannot find {CLASSES_PATH}")
        print("Run 'ls data/' to see exactly what is in the main data folder.")