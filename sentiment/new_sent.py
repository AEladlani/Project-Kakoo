import streamlit as st
from PIL import Image
import string
import re
from collections import Counter
import matplotlib.pyplot as plt

from deepmultilingualpunctuation import PunctuationModel
from transformers import pipeline
from mediapipe.tasks import python
from mediapipe.tasks.python import text as mp_text

# ---------------------------
# 1️⃣ Streamlit UI
# ---------------------------
st.set_page_config(page_title="Kakoo Sentiment Analyzer", layout="wide")

# Show Kakoo logo
logo = Image.open("kakoo.png")
st.image(logo, width=200)
st.title("Kakoo HR Interview Sentiment Analyzer")

# ---------------------------
# 2️⃣ Choose input method
# ---------------------------
input_method = st.radio(
    "How do you want to provide the text?",
    ("Write/Paste Text", "Upload File"))

text_input = ""

if input_method == "Write/Paste Text":
    text_input = st.text_area("Paste your interview answer here:")

elif input_method == "Upload File":
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded_file is not None:
        text_input = uploaded_file.read().decode("utf-8")

# Only proceed if text is provided
if text_input:

    st.markdown("### 1️⃣ Detect Language")
    # ---------------------------
    # Language Detection
    # ---------------------------
    model_path = 'det_lang_mod/language_detector.tflite'
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = mp_text.LanguageDetectorOptions(base_options=base_options)

    with mp_text.LanguageDetector.create_from_options(options) as detector:
        detection_result = detector.detect(text_input)

    detection = detection_result.detections[0]
    language_code = detection.language_code
    confidence = detection.probability

    if language_code == 'en':
        lang = 'English'
    elif language_code == 'fr':
        lang = 'Français'
    else:
        lang = f"Unknown Language ({language_code})"

    st.write(f"Language: **{lang}** (Confidence: {confidence:.2f})")
    st.write("---------------------------------------------------------------------------")
    # ---------------------------
    # 2️⃣ Punctuation Restoration
    # ---------------------------
    st.markdown("### 2️⃣ Punctuation Restoration")
    punct_model = PunctuationModel()
    def remove_punctuation(text):
        return text.translate(str.maketrans('', '', string.punctuation))
    def restore_punctuation(paragraph):
        text_no_punct = remove_punctuation(paragraph)
        punctuated_text = punct_model.restore_punctuation(text_no_punct)
        return punctuated_text
    st.subheader("Original Text")
    st.write(text_input)
    st.write("---------------------------------------------------------------------------")
    restored_paragraph = restore_punctuation(text_input)
    st.subheader("Punctuated Text")
    st.write(restored_paragraph)
    # ---------------------------
    # 3️⃣ Split into Sentences
    # ---------------------------
    st.markdown("### 3️⃣ Split into Sentences")
    def split_sentences(text):
        sentences = re.split(r'(?<=[.!?]) +', text)
        return [s.strip() for s in sentences if s.strip()]
    sentences = split_sentences(restored_paragraph)
    st.write(sentences)
    st.write("---------------------------------------------------------------------------")
    # ---------------------------
    # 4️⃣ Sentiment Analysis
    # ---------------------------
    st.markdown("### 4️⃣ Sentiment Analysis")
    sentiment_model = pipeline("sentiment-analysis",
                               model="clapAI/mmBERT-small-multilingual-sentiment")
    def analyze_sentiment(text):
        result = sentiment_model(text)[0]
        label = result["label"]
        if label.startswith("LABEL_"):
            LABEL_MAP = {
                "LABEL_0": "negative",
                "LABEL_1": "neutral",
                "LABEL_2": "positive"}
            sentiment = LABEL_MAP[label]
        else:
            sentiment = label.lower()
        confidence = result["score"]
        return sentiment, confidence
        
    def color_text(text, sentiment):
            colors = {"positive": "blue", "neutral": "green", "negative": "red"}
            st.markdown(f"<span style='color:{colors[sentiment]}'>{text}</span>", unsafe_allow_html=True)

    predictions = []
    #st.subheader("Sentence-level Sentiment")
    for s in sentences:
        sentiment, confidence = analyze_sentiment(s)
        predictions.append(sentiment)
        #st.write(f"{s} → **{sentiment}** ({confidence:.2f})")
        output = f"{s} → {sentiment} ({confidence:.2f})"
        color_text(output, sentiment)
    st.write("---------------------------------------------------------------------------")
   
    # ---------------------------
    # 5️⃣ Pie Chart of Sentiment
    # ---------------------------
    st.markdown("### 5️⃣ Sentiment Distribution Pie Chart")
    counts = Counter(predictions)
    labels = list(counts.keys())
    sizes = list(counts.values())
    colors = ['#ff9999','#66b3ff','#99ff99']

    fig, ax = plt.subplots(figsize=(5,5))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title("Sentiment Distribution")
    st.pyplot(fig)