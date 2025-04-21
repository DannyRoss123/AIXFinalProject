import streamlit as st
import pandas as pd
import openai
import os
import textstat
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import torch
from sentence_transformers import SentenceTransformer, util
from xgboost import XGBClassifier
import joblib
from openai import OpenAI

# --- NLP + Embedding Setup ---
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

# Fix for PyTorch compatibility issue
# Try to use CPU device to avoid the meta tensor error
device = torch.device("cpu")
try:
    model = SentenceTransformer("all-MiniLM-L6-v2").to(device)
except NotImplementedError:
    # If that fails, try this alternative approach
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# --- GPT API Setup ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"]) 

# --- Load Your Model ---
clf = joblib.load("xgb_model.pkl")

# --- Extract Features for Model ---
def extract_features(prompt, resp_a, resp_b):
    # Convert to tensor with CPU device explicitly
    prompt_emb = model.encode(prompt, convert_to_tensor=True, device=device)
    emb_a = model.encode(resp_a, convert_to_tensor=True, device=device)
    emb_b = model.encode(resp_b, convert_to_tensor=True, device=device)

    sim_a = util.cos_sim(prompt_emb, emb_a).item()
    sim_b = util.cos_sim(prompt_emb, emb_b).item()

    len_a, len_b = len(resp_a), len(resp_b)
    word_a, word_b = len(resp_a.split()), len(resp_b.split())
    read_a = textstat.flesch_reading_ease(resp_a)
    read_b = textstat.flesch_reading_ease(resp_b)
    sent_a = sia.polarity_scores(resp_a)["compound"]
    sent_b = sia.polarity_scores(resp_b)["compound"]

    # Normalize length features to reduce their impact
    relative_length_diff = (len_a - len_b) / (len_a + len_b) if (len_a + len_b) > 0 else 0

    return pd.DataFrame([{
        "prompt_similarity_diff": sim_a - sim_b,
        "readability_diff": read_a - read_b,
        "length_diff": relative_length_diff,
        "sentiment_diff": sent_a - sent_b
    }]), {
        "Prompt Similarity": round(sim_a - sim_b, 3),
        "Readability": round(read_a - read_b, 3),
        "Length (normalized)": round(relative_length_diff, 3),
        "Sentiment": round(sent_a - sent_b, 3),
        "Raw Length A": len_a,
        "Raw Length B": len_b,
        "Raw Similarity A": round(sim_a, 3),
        "Raw Similarity B": round(sim_b, 3),
        "Raw Readability A": round(read_a, 3),
        "Raw Readability B": round(read_b, 3)
    }

# --- Generate GPT Responses ---
def generate_responses(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Give me two different answers to the following question:\n\n{prompt}"}
        ],
        temperature=0.7
    )
    content = response.choices[0].message.content.strip()
    responses = content.split("\n\n")
    if len(responses) >= 2:
        return responses[0].strip(), responses[1].strip()
    else:
        return content.strip(), "GPT did not return a second distinct response."

# --- Streamlit App ---
st.set_page_config(page_title="GPT Response Preference Predictor", layout="centered")
st.title("🧠 GPT Response Comparator")
st.markdown("Enter a prompt. GPT will generate two responses. Your model will predict which one you prefer and explain why.")

prompt = st.text_area("📝 Enter your GPT prompt", height=120)

if st.button("✨ Generate Responses and Predict"):
    if prompt:
        with st.spinner("Generating GPT responses..."):
            resp_a, resp_b = generate_responses(prompt)

        st.subheader("🅰️ Response A")
        st.markdown(f"> {resp_a}")
        st.subheader("🅱️ Response B")
        st.markdown(f"> {resp_b}")

        with st.spinner("Analyzing responses..."):
            features, scores = extract_features(prompt, resp_a, resp_b)
            
            # Get prediction and probabilities
            prediction = clf.predict(features)[0]  # 1 means A preferred
            probabilities = clf.predict_proba(features)[0]
            confidence = probabilities[1] if prediction == 1 else probabilities[0]
            
            preferred = "🅰️ Response A" if prediction == 1 else "🅱️ Response B"

        st.success(f"✅ It is predicted that you prefer **{preferred}** with {confidence:.2%} confidence.")

        st.markdown("---")
        st.subheader("📊 Feature Differences")
        
        # Display feature differences
        feature_df = pd.DataFrame({
            "Feature": ["Prompt Similarity", "Readability", "Length (normalized)", "Sentiment"],
            "Value": [
                scores["Prompt Similarity"],
                scores["Readability"],
                scores["Length (normalized)"],
                scores["Sentiment"]
            ]
        })
        st.table(feature_df)

        # Show raw values for comparison
        with st.expander("Show Raw Feature Values"):
            raw_df = pd.DataFrame({
                "Feature": ["Similarity", "Readability", "Length", "Sentiment"],
                "Response A": [
                    scores["Raw Similarity A"],
                    scores["Raw Readability A"],
                    scores["Raw Length A"],
                    round(sia.polarity_scores(resp_a)["compound"], 3)
                ],
                "Response B": [
                    scores["Raw Similarity B"],
                    scores["Raw Readability B"],
                    scores["Raw Length B"],
                    round(sia.polarity_scores(resp_b)["compound"], 3)
                ]
            })
            st.table(raw_df)

        st.markdown("### 🔍 Why the Model Prefers This Response")

        # Only explain the positive reasons for the chosen response
        chosen_letter = "A" if prediction == 1 else "B"
        reasons = []

        sim = scores["Prompt Similarity"]
        read = scores["Readability"]
        length = scores["Length (normalized)"]
        sent = scores["Sentiment"]

        # Always explain in terms of the chosen response's advantages
        if prediction == 1:  # A is preferred
            if sim > 0.05:
                reasons.append("- Has better semantic alignment with your prompt")
            if read > 5:
                reasons.append("- Is easier to read with better readability")
            if length > 0.1:
                reasons.append("- Provides more detailed information")
            if sent > 0.1:
                reasons.append("- Has a more positive or helpful tone")
        else:  # B is preferred
            if sim < -0.05:
                reasons.append("- Has better semantic alignment with your prompt")
            if read < -5:
                reasons.append("- Is easier to read with better readability")
            if length < -0.1:
                reasons.append("- Provides more detailed information")
            if sent < -0.1:
                reasons.append("- Has a more positive or helpful tone")

        if reasons:
            st.markdown(f"Response {chosen_letter} was chosen because it:")
            st.markdown("\n".join(reasons))
        else:
            st.markdown(f"Response {chosen_letter} was chosen based on small advantages across multiple features.")
    else:
        st.warning("Please enter a prompt.")
