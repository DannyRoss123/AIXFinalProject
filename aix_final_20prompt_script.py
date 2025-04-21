
import openai
import pandas as pd

# === STEP 1: OpenAI API Key ===
client = openai.OpenAI(api_key="key here")
# === STEP 2: 170 Clear, Distinct Prompts ===
prompts = [
    "Why is the sky blue?",
    "How do airplanes fly?",
    "What causes earthquakes?",
    "Explain gravity in simple terms.",
    "Why do we need sleep?",
    "What is climate change?",
    "How do vaccines work?",
    "What causes rainbows?",
    "Why do cats purr?",
    "How does a car engine work?",
    "What is artificial intelligence?",
    "How does the internet work?",
    "What is cryptocurrency?",
    "Explain black holes.",
    "How do plants grow?",
    "What is the greenhouse effect?",
    "How does a microwave work?",
    "Why is the ocean salty?",
    "How does photosynthesis work?",
    "What causes lightning?",
    "Why do seasons change?",
    "How do computers store information?",
    "What is machine learning?",
    "How does 3D printing work?",
    "Why do we dream?",
    "How does the human brain work?",
    "What is DNA?",
    "How do magnets work?",
    "What is the Big Bang?",
    "What is a quantum computer?",
    "What is inflation in economics?",
    "How does a credit card work?",
    "How do planes stay in the air?",
    "What is the stock market?",
    "What causes tsunamis?",
    "Why is the sky sometimes red?",
    "What is democracy?",
    "Why do we cry?",
    "How do electric cars work?",
    "What is nuclear energy?",
    "How do cell phones work?",
    "What is renewable energy?",
    "What causes tides?",
    "What is a solar eclipse?",
    "What is a lunar eclipse?",1
    "How does GPS work?",
    "What is an algorithm?",
    "How does Wi-Fi work?",
    "How does Bluetooth work?",
    "What is a black hole?",
    "Why do we have eyebrows?",
    "How does sound travel?",
    "What is a virus?",
    "How do antibiotics work?",
    "What is the immune system?",
    "How does the heart work?",
    "What is cholesterol?",
    "How do lungs function?",
    "What is artificial selection?",
    "What causes obesity?",
    "How do mirrors work?",
    "What is dark matter?",
    "What is antimatter?",
    "How do rainforests help the planet?",
    "What is biodiversity?",
    "How do bees make honey?",
    "How does digestion work?",
    "What is evolution?",
    "How does recycling help?",
    "What causes climate change?",
    "Why are pandas endangered?",
    "What is the ozone layer?",
    "How do glasses help vision?",
    "Why is exercise important?",
    "What is yoga?",
    "How does meditation help the brain?",
    "What is mindfulness?",
    "Why do people get addicted?",
    "What is depression?",
    "How does therapy work?",
    "What is empathy?",
    "What is IQ?",
    "How do lie detectors work?",
    "Why do we laugh?",
    "What is music therapy?",
    "What are dreams?",
    "How does caffeine work?",
    "How does alcohol affect the brain?",
    "What is dopamine?",
    "How does memory work?",
    "Why do we forget things?",
    "What is anxiety?",
    "How does the placebo effect work?",
    "What is neuroscience?",
    "What is psychology?",
    "What is philosophy?",
    "How does logic work?",
    "What is ethics?",
    "What is consciousness?",
    "How does perception work?",
    "Why do we see optical illusions?",
]

# === STEP 3: Generate 2 Responses Per Prompt and Manually Label ===
data = []

def get_responses(prompt):
    responses = []
    for _ in range(2):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9
        )
        responses.append(response.choices[0].message.content.strip())
    return responses

for prompt in prompts:
    print(f"\n\n📌 PROMPT: {prompt}")
    responses = get_responses(prompt)

    for i, res in enumerate(responses):
        print(f"\n--- Response {i} ---\n{res}\n")

    while True:
        try:
            best = int(input("Which response is best? (0 or 1): "))
            if best in [0, 1]:
                break
            else:
                print("❗ Please enter 0 or 1.")
        except ValueError:
            print("❗ Invalid input. Enter a number.")

    for i, res in enumerate(responses):
        data.append({
            "prompt": prompt,
            "response": res,
            "variant": i,
            "is_best": 1 if i == best else 0
        })

# === STEP 4: Save to CSV ===
df = pd.DataFrame(data)
df.to_csv("labeled_responses_100x2.csv", index=False)
print("\n✅ CSV saved as labeled_responses_170x2.csv")
