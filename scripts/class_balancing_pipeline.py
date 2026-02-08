import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

def run_balancing_protocol(classes_path, features_path):
    # 1. Ingestion & Merging
    classes = pd.read_csv(classes_path)
    features = pd.read_csv(features_path, header=None)
    features.columns = ['txId', 'time_step'] + [f'feat_{i}' for i in range(1, 166)]
    classes['class'] = classes['class'].replace({'unknown': '3'}).astype(int)
    df = pd.merge(features, classes, on='txId')

    labeled_df = df[df['class'].isin([1, 2])].copy()
    labeled_df['label'] = labeled_df['class'].apply(lambda x: 1 if x == 1 else 0)

    # 2. Temporal Split (Consistency is key for the Audit)
    train_df = labeled_df[labeled_df['time_step'] <= 34]
    test_df  = labeled_df[labeled_df['time_step'] > 34]

    X_train = train_df.drop(['txId', 'class', 'label', 'time_step'], axis=1)
    y_train = train_df['label']
    X_test = test_df.drop(['txId', 'class', 'label', 'time_step'], axis=1)
    y_test = test_df['label']

    print(f"Pre-SMOTE Training Count: {len(X_train)} (Illicit: {sum(y_train)})")

    # 3. SMOTE Application
    # Synthesizing samples to balance the forensic minority class
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print(f"Post-SMOTE Training Count: {len(X_train_res)} (Illicit: {sum(y_train_res)})")

    # 4. Re-Evaluation
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_res, y_train_res)
    
    return y_test, clf.predict(X_test)

if __name__ == "__main__":
    ROOT_DATA = 'data/'
    y_true, y_pred = run_balancing_protocol(
        f'{ROOT_DATA}elliptic_txs_classes.csv', 
        f'{ROOT_DATA}elliptic_txs_features.csv'
    )
    
    print(f"\n{'='*40}\nSMOTE-ENHANCED PERFORMANCE REPORT\n{'='*40}")
    print(classification_report(y_true, y_pred, target_names=['Licit (0)', 'Illicit (1)']))