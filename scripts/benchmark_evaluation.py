import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def run_benchmark_audit(classes_path, features_path):
    # 1. Load and Merging
    classes = pd.read_csv(classes_path)
    features = pd.read_csv(features_path, header=None)
    
    # Standardize column naming
    features.columns = ['txId', 'time_step'] + [f'feat_{i}' for i in range(1, 166)]
    classes['class'] = classes['class'].replace({'unknown': '3'}).astype(int)
    
    df = pd.merge(features, classes, on='txId')
    
    # 2. Filter for Labeled Data (Class 1 or 2)
    labeled_df = df[df['class'].isin([1, 2])].copy()
    labeled_df['label'] = labeled_df['class'].apply(lambda x: 1 if x == 1 else 0)

    # 3. Temporal Split (Industry Standard)
    # Train on past (1-34), Test on future (35+)
    train_df = labeled_df[labeled_df['time_step'] <= 34]
    test_df  = labeled_df[labeled_df['time_step'] > 34]

    X_train = train_df.drop(['txId', 'class', 'label', 'time_step'], axis=1)
    y_train = train_df['label']
    X_test = test_df.drop(['txId', 'class', 'label', 'time_step'], axis=1)
    y_test = test_df['label']

    print(f"Audit Scope: {len(X_train)} Training Samples | {len(X_test)} Test Samples")

    # 4. Standard Random Forest (No Balancing)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    return y_test, clf.predict(X_test)

if __name__ == "__main__":
    ROOT_DATA = 'data/'
    y_true, y_pred = run_benchmark_audit(
        f'{ROOT_DATA}elliptic_txs_classes.csv', 
        f'{ROOT_DATA}elliptic_txs_features.csv'
    )
    
    print(f"\n{'='*40}\nBASELINE PERFORMANCE REPORT\n{'='*40}")
    print(classification_report(y_true, y_pred, target_names=['Licit (0)', 'Illicit (1)']))