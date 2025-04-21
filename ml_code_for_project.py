import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import textstat
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, util
from sklearn.utils import resample
from xgboost import XGBClassifier, plot_tree

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
model = SentenceTransformer("all-MiniLM-L6-v2")


def load_and_pair_data(filepath):
    print("[1] Loading and pairing data...")
    df = pd.read_csv(filepath)

    pair_rows = []
    for prompt, group in df.groupby("prompt"):
        responses = group.sort_values("variant")
        if len(responses) == 2:
            r0, r1 = responses.iloc[0], responses.iloc[1]
            label = 1 if r0['is_best'] == 1 else 0
            pair_rows.append({
                "prompt": prompt,
                "response_a": r0['response'],
                "response_b": r1['response'],
                "label": label
            })
    return pd.DataFrame(pair_rows)


def extract_pairwise_features(df):
    print("[2] Extracting features...")
    features = []

    for _, row in df.iterrows():
        prompt = row["prompt"]
        ra = row["response_a"]
        rb = row["response_b"]

        # Embeddings
        prompt_emb = model.encode(prompt, convert_to_tensor=True)
        emb_a = model.encode(ra, convert_to_tensor=True)
        emb_b = model.encode(rb, convert_to_tensor=True)

        sim_a = util.cos_sim(prompt_emb, emb_a).item()
        sim_b = util.cos_sim(prompt_emb, emb_b).item()

        # Readability, style, sentiment
        len_a, len_b = len(ra), len(rb)
        word_a, word_b = len(ra.split()), len(rb.split())
        read_a = textstat.flesch_reading_ease(ra)
        read_b = textstat.flesch_reading_ease(rb)
        sent_a = sia.polarity_scores(ra)["compound"]
        sent_b = sia.polarity_scores(rb)["compound"]

        # Normalize length features to reduce their impact
        relative_length_diff = (len_a - len_b) / (len_a + len_b)
        relative_word_count_diff = (word_a - word_b) / (word_a + word_b)

        features.append({
            "prompt_similarity_diff": sim_a - sim_b,
            "readability_diff": read_a - read_b,
            "length_diff":  relative_length_diff,
            "sentiment_diff": sent_a - sent_b,
            "label": row["label"]
        })

    return pd.DataFrame(features)


def train_model(features_df):
    print("[3] Training model with constrained feature weights...")

    X = features_df.drop("label", axis=1)
    y = features_df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Create weight constraints for features
    # This significantly reduces the influence of the length feature
    feature_weights = {
        'prompt_similarity_diff': 1.0,
        'readability_diff': 1.0,
        'length_diff': 0.2,  # Reduce the weight of length feature
        'sentiment_diff': 1.0
    }
    
    # Configure XGBoost with feature weights
    clf = XGBClassifier(
        use_label_encoder=True, 
        eval_metric='logloss', 
        random_state=42,
        # Add other parameters to reduce the importance of specific features
        max_depth=3,  # Reduce tree depth to avoid overfitting to length
        colsample_bytree=0.8,  # Subsample columns for each tree
        subsample=0.8  # Subsample observations for each tree
    )
    
    # Create sample weights that reduce the influence of examples where length difference is large
    # This makes the model focus less on length as a predictor
    length_diffs = abs(X_train['length_diff'])
    sample_weights = 1 / (1 + 2 * length_diffs)  # Higher weight for examples with similar lengths
    sample_weights = sample_weights / sample_weights.mean()  # Normalize

    # Train with sample weights
    clf.fit(X_train, y_train, sample_weight=sample_weights)

    print("[4] Evaluating model...")
    y_pred = clf.predict(X_test)
    print(y_pred)
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["B Better", "A Better"], yticklabels=["B Better", "A Better"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("xgboost_confusion_matrix.png")
    plt.show()

    # Feature importance
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": clf.feature_importances_
    }).sort_values(by="Importance", ascending=True)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=importance_df, x="Importance", y="Feature")
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig("xgboost_feature_importance.png")
    plt.show()

    import joblib
    joblib.dump(clf, "xgb_model.pkl")

    return clf


def predict_preference(prompt, resp_a, resp_b, clf):
    prompt_emb = model.encode(prompt, convert_to_tensor=True)
    emb_a = model.encode(resp_a, convert_to_tensor=True)
    emb_b = model.encode(resp_b, convert_to_tensor=True)

    sim_a = util.cos_sim(prompt_emb, emb_a).item()
    sim_b = util.cos_sim(prompt_emb, emb_b).item()

    len_a, len_b = len(resp_a), len(resp_b)
    word_a, word_b = len(resp_a.split()), len(resp_b.split())
    read_a = textstat.flesch_reading_ease(resp_a)
    read_b = textstat.flesch_reading_ease(resp_b)
    sent_a = sia.polarity_scores(resp_a)["compound"]
    sent_b = sia.polarity_scores(resp_b)["compound"]

    # Normalize length features to reduce their impact
    relative_length_diff = (len_a - len_b) / (len_a + len_b)
    relative_word_count_diff = (word_a - word_b) / (word_a + word_b)

    sample = pd.DataFrame([{
        "prompt_similarity_diff": sim_a - sim_b,
        "readability_diff": read_a - read_b,
        "length_diff":  relative_length_diff,
        "sentiment_diff": sent_a - sent_b
    }])

    # Print feature values for debugging
    print("Feature values:")
    for col in sample.columns:
        print(f"  {col}: {sample[col].values[0]:.3f}")
    
    # Get prediction and probabilities
    prediction = clf.predict(sample)[0]
    probabilities = clf.predict_proba(sample)[0]
    print(f"Prediction: {prediction} (0=B better, 1=A better)")
    print(f"Probabilities: B={probabilities[0]:.3f}, A={probabilities[1]:.3f}")

    return "Response A is better" if prediction == 1 else "Response B is better"


# === MAIN ===
def main():
    print("=== GPT Output Preference Classifier v2 (XGBoost) ===")

    pairs_df = load_and_pair_data("labeled_responses_100x2.csv")
    features_df = extract_pairwise_features(pairs_df)
    clf = train_model(features_df)

    print("\n[6] Example prediction...")
    prompt = "Why is the sky blue?"
    response_a = "The sky appears blue due to Rayleigh scattering in the atmosphere."
    response_b = "The reason the sky is blue is because the sun shines and that's what happens."

    result = predict_preference(prompt, response_a, response_b, clf)
    print(f"\nPrompt: {prompt}\nPrediction: {result}")

if __name__ == "__main__":
    main()