import streamlit as st
import pandas as pd
import os
import textstat
import nltk
import torch
import joblib
import re
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
from xgboost import XGBClassifier

# --- Setup ---
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()
device = torch.device("cpu")

try:
    model = SentenceTransformer("all-MiniLM-L6-v2").to(device)
except NotImplementedError:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
clf = joblib.load("xgb_model.pkl")

# --- Feature Extraction ---
def extract_features(prompt, resp_a, resp_b):
    prompt_emb = model.encode(prompt, convert_to_tensor=True, device=device)
    emb_a = model.encode(resp_a, convert_to_tensor=True, device=device)
    emb_b = model.encode(resp_b, convert_to_tensor=True, device=device)

    sim_a = util.cos_sim(prompt_emb, emb_a).item()
    sim_b = util.cos_sim(prompt_emb, emb_b).item()
    len_a, len_b = len(resp_a), len(resp_b)
    read_a, read_b = textstat.flesch_reading_ease(resp_a), textstat.flesch_reading_ease(resp_b)
    sent_a, sent_b = sia.polarity_scores(resp_a)["compound"], sia.polarity_scores(resp_b)["compound"]

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

# --- GPT Generation ---
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

    # Try to split on numbered format
    numbered = re.split(r"\n?\s*1\.\s*", content)
    if len(numbered) > 1:
        parts = re.split(r"\n?\s*2\.\s*", numbered[1])
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()

    # Fallback: double newline
    splits = content.split("\n\n")
    if len(splits) >= 2:
        return splits[0].strip(), splits[1].strip()

    # Last resort: single newline
    lines = content.split("\n")
    if len(lines) >= 2:
        return lines[0].strip(), lines[1].strip()

    return content.strip(), "GPT did not return a second distinct response."

# --- Streamlit App ---
st.set_page_config(page_title="GPT Response Preference Predictor", layout="centered")
st.title("\U0001F9E0 GPT Response Comparator")
st.markdown("Enter a prompt. GPT will generate two responses. Your model will predict which one you prefer and explain why.")

prompt = st.text_area("\U0001F4DD Enter your GPT prompt", height=120)

if st.button("\u2728 Generate Responses and Predict"):
    if prompt:
        with st.spinner("Generating GPT responses..."):
            resp_a, resp_b = generate_responses(prompt)

        st.subheader("\U0001F170️ Response A")
        st.markdown(f"> {resp_a}")
        st.subheader("\U0001F171️ Response B")
        st.markdown(f"> {resp_b}")

        if "GPT did not return" in resp_b:
            st.warning("⚠️ GPT only returned one usable response. Try rephrasing your prompt or click again.")
        else:
            with st.spinner("Analyzing responses..."):
                features, scores = extract_features(prompt, resp_a, resp_b)
                prediction = clf.predict(features)[0]
                probabilities = clf.predict_proba(features)[0]
                confidence = probabilities[1] if prediction == 1 else probabilities[0]
                preferred = "\U0001F170️ Response A" if prediction == 1 else "\U0001F171️ Response B"

            st.success(f"✅ It is predicted that you prefer **{preferred}** with {confidence:.2%} confidence.")

            st.markdown("---")
            st.subheader("\U0001F4CA Feature Differences")
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

            st.markdown("### \U0001F50D Why the Model Prefers This Response")
            chosen_letter = "A" if prediction == 1 else "B"
            sim, read, length, sent = scores["Prompt Similarity"], scores["Readability"], scores["Length (normalized)"], scores["Sentiment"]

            reasons = []
            if prediction == 1:
                if sim > 0.05: reasons.append("- Has better semantic alignment with your prompt")
                if read > 5: reasons.append("- Is easier to read with better readability")
                if length > 0.1: reasons.append("- Provides more detailed information")
                if sent > 0.1: reasons.append("- Has a more positive or helpful tone")
            else:
                if sim < -0.05: reasons.append("- Has better semantic alignment with your prompt")
                if read < -5: reasons.append("- Is easier to read with better readability")
                if length < -0.1: reasons.append("- Provides more detailed information")
                if sent < -0.1: reasons.append("- Has a more positive or helpful tone")

            if reasons:
                st.markdown(f"Response {chosen_letter} was chosen because it:")
                st.markdown("\n".join(reasons))
            else:
                st.markdown(f"Response {chosen_letter} was chosen based on subtle advantages across multiple features.")
    else:
        st.warning("Please enter a prompt.")
