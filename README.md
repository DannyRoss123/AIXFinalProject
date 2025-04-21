# GPT Response Comparator

This project is a web application built using Streamlit that predicts which of two GPT-generated responses a user is more likely to prefer. The core idea behind this tool is to explore how machine learning can be used to evaluate natural language responses in a more human-centric way, by identifying subtle differences in writing quality, relevance, and tone.

When a user enters a prompt, the app uses the OpenAI API to generate two different responses. These responses are then analyzed using a variety of features, including their semantic similarity to the prompt, their readability scores, length, and sentiment polarity. A trained XGBoost model then evaluates these features to predict which response the user would likely find more appropriate or helpful.

The machine learning model was trained on a custom-labeled dataset consisting of 100 prompts that I hand-labeled. For each prompt, I generated two GPT responses and manually selected the one I preferred. These preferences were encoded as binary labels and used to train the model. Feature extraction included sentence embedding comparisons using the Sentence Transformers library, Flesch readability scores via the TextStat library, and sentiment analysis through NLTK's VADER sentiment analyzer.

The project consists of several core files. The main application logic is contained in `create_website.py`, which includes the Streamlit interface and the logic to generate GPT responses, extract features, and display predictions. The trained XGBoost model is saved as `xgb_model.pkl`, and is loaded during runtime to make predictions. A `requirements.txt` file is included to specify the Python dependencies necessary to run the application. Additionally, the OpenAI API key is securely loaded from a file named `secrets.toml` inside a `.streamlit` directory, which is intentionally excluded from version control for security purposes.

This project was designed both as a technical exercise and as an exploration into how preference modeling can be applied to AI outputs. While the dataset is small and the model is limited, it demonstrates how explainable, feature-based evaluation of AI-generated content can be made accessible through a simple user interface. It also highlights the potential of using learned preference signals as an alternative to more rigid evaluation methods when comparing model outputs.

Through building this project, I gained hands-on experience with end-to-end machine learning workflows, including data labeling, feature engineering, model training, and deploying an interactive app to the web. I also became more comfortable working with APIs, NLP feature extraction techniques, and tools like Streamlit for rapid prototyping. The project deepened my understanding of how subtle metrics like sentiment, readability, and semantic alignment can inform real-world ML applications.

In the future, this project could be improved in several ways. One clear direction is expanding the dataset — with more hand-labeled comparisons, the model would likely become more accurate and generalizable. Additionally, incorporating more sophisticated NLP features (such as coherence, specificity, or engagement scoring) could make the preference prediction more robust. It would also be interesting to allow users to vote on which response they actually prefer and use that feedback to continuously retrain the model, making it a live reinforcement learning system based on real user input.

To run the app locally, users should clone the repository, install the required dependencies listed in `requirements.txt`, and supply their own OpenAI API key by creating a `.streamlit/secrets.toml` file with the following structure:

OPENAI_API_KEY = "your-api-key-here"

Once set up, the app can be run using the command `streamlit run create_website.py`. If deployed to Streamlit Cloud, the app can be accessed via a public URL.

This project reflects my continued interest in AI alignment, interpretability, and interactive tools that allow users to better understand how models make decisions.
