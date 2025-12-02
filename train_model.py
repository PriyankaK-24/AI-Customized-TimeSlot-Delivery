# train_model.py
"""
Train a simple slot-prediction model and save it to slot_model.joblib

Run:
    python train_model.py
"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def load_data(path='deliveries.csv'):
    df = pd.read_csv(path)
    return df

def prepare_features(df):
    # Simple feature set for MVP
    # Keep numeric columns only; encode day-of-week if needed
    # We'll one-hot encode dow for a slightly better model
    X = df[['customer_id','work','attempts_before','hour_pref_provided']].copy()
    # one-hot encode day-of-week
    dow_dummies = pd.get_dummies(df['dow'], prefix='dow')
    X = pd.concat([X, dow_dummies], axis=1)
    y = df['timeslot']
    return X, y

def train_and_save(X, y, out_file='slot_model.joblib'):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        stratify=y_enc)
    clf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    print("Training model...")
    clf.fit(X_train, y_train)
    print("Predicting on test set...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    # Save model + label encoder
    joblib.dump({'model': clf, 'le': le, 'features': X.columns.tolist()}, out_file)
    print(f"Saved model to {out_file}")

def quick_demo(out_file='slot_model.joblib'):
    # Quick load and demo one prediction
    d = joblib.load(out_file)
    clf = d['model']
    le = d['le']
    features = d['features']
    import numpy as np
    # demo input (random plausible)
    sample = {
        'customer_id': 12,
        'work': 0,
        'attempts_before': 0,
        'hour_pref_provided': 1,
        # add dow one-hot columns (set Mon=1 example)
    }
    # build input vector with same columns
    row = []
    for f in features:
        if f in sample:
            row.append(sample[f])
        elif f.startswith('dow_'):
            # default all zeros, set 'dow_Mon' to 1 for demo
            row.append(1 if f == 'dow_Mon' else 0)
        else:
            row.append(0)
    prob = clf.predict_proba([row])[0]
    idx = prob.argmax()
    print("Demo suggested slot:", le.inverse_transform([idx])[0])
    # show top-3 probs
    top_idx = np.argsort(prob)[::-1][:3]
    print("Top slot probabilities:")
    for i in top_idx:
        print(f"  {le.inverse_transform([i])[0]} : {prob[i]:.2f}")

if __name__ == "__main__":
    df = load_data('deliveries.csv')
    X, y = prepare_features(df)
    train_and_save(X, y, 'slot_model.joblib')
    quick_demo('slot_model.joblib')
