import os
import re
import string
import joblib
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sentify",
    layout="centered"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

    /* Main page */
    .stApp {
        background-color: #E0F2FE;
    }

    /* Main container */
    .block-container {
        max-width: 850px;
        padding-top: 3rem;
        padding-bottom: 2rem;
    }

    /* Title */
    .title {
        text-align: center;
        color: #0369A1;
        font-size: 48px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #475569;
        font-size: 18px;
        margin-bottom: 30px;
    }

    /* Text area */
    textarea {
        border: 2px solid #7DD3FC !important;
        border-radius: 12px !important;
        background-color: white !important;
        color: #0F172A !important;
        font-size: 16px !important;
    }

    textarea:focus {
        border: 2px solid #0284C7 !important;
        box-shadow: 0 0 8px rgba(2, 132, 199, 0.2) !important;
    }

    /* Analyze button */
    .stButton > button {
        width: 100%;
        background-color: #0284C7;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 20px;
        font-size: 17px;
        font-weight: 600;
        transition: 0.2s;
    }

    .stButton > button:hover {
        background-color: #0369A1;
        color: white;
    }

    /* Example section */
    .example-title {
        color: #0369A1;
        font-size: 20px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 10px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748B;
        font-size: 14px;
        margin-top: 40px;
    }

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_PATH = "models/sentiment_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    # Verify fitted TF-IDF vectorizer
    if not hasattr(vectorizer, "idf_"):
        raise ValueError(
            "The TF-IDF vectorizer is not fitted."
        )

    if not hasattr(vectorizer, "vocabulary_"):
        raise ValueError(
            "The TF-IDF vectorizer has no vocabulary."
        )

    return model, vectorizer


# ============================================================
# LOAD MODEL SAFELY
# ============================================================

try:

    model, vectorizer = load_model()

except FileNotFoundError:

    st.error(
        "Model files not found. "
        "Please run the Jupyter Notebook first."
    )

    st.stop()

except Exception as e:

    st.error(
        f"Error loading model: {e}"
    )

    st.stop()


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    # Remove mentions
    text = re.sub(
        r"@\w+",
        "",
        text
    )

    # Remove hashtag symbol
    text = re.sub(
        r"#",
        "",
        text
    )

    # Remove HTML tags
    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    # Remove punctuation
    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    # Remove numbers
    text = re.sub(
        r"\d+",
        "",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_sentiment(text):

    cleaned_text = clean_text(text)

    text_vector = vectorizer.transform(
        [cleaned_text]
    )

    prediction = model.predict(
        text_vector
    )[0]

    return (
        str(prediction).lower().strip(),
        cleaned_text
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">💬 Sentify</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
<div class="subtitle">
    Social Media Sentiment Analysis
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# INTRODUCTION
# ============================================================

st.markdown(
    """
<div style="
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 20px;
    color: #334155;
    text-align: center;
    box-shadow: 0 3px 10px rgba(15,23,42,0.06);
">
    Enter a social media post below and Sentify will
    classify its sentiment as <b>Positive</b>,
    <b>Negative</b>, or <b>Neutral</b>.
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "example_text" not in st.session_state:

    st.session_state.example_text = ""


# ============================================================
# TEXT INPUT
# ============================================================

text = st.text_area(
    "Enter your text",
    value=st.session_state.example_text,
    height=150,
    placeholder="Example: I absolutely love this product!"
)


# ============================================================
# EXAMPLES
# ============================================================

st.markdown(
    '<div class="example-title">Moods</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


with col1:

    if st.button("😊 Positive"):

        st.session_state.example_text = (
            "I absolutely love this app! "
            "It is amazing and very useful."
        )

        st.rerun()


with col2:

    if st.button("😐 Neutral"):

        st.session_state.example_text = (
            "The application was released yesterday "
            "with several new features."
        )

        st.rerun()


with col3:

    if st.button("😞 Negative"):

        st.session_state.example_text = (
            "This app is terrible and completely useless."
        )

        st.rerun()


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.write("")

if st.button("🔍 Analyze Sentiment"):

    if not text.strip():

        st.warning(
            "Please enter some text before analyzing."
        )

    else:

        prediction, cleaned_text = predict_sentiment(text)


        # ====================================================
        # POSITIVE
        # ====================================================

        if prediction == "positive":

            st.success(
                "😊 POSITIVE SENTIMENT"
            )

            st.write(
                "The text expresses a positive sentiment."
            )


        # ====================================================
        # NEGATIVE
        # ====================================================

        elif prediction == "negative":

            st.error(
                "😞 NEGATIVE SENTIMENT"
            )

            st.write(
                "The text expresses a negative sentiment."
            )


        # ====================================================
        # NEUTRAL
        # ====================================================

        elif prediction == "neutral":

            st.info(
                "😐 NEUTRAL SENTIMENT"
            )

            st.write(
                "The text appears to be neutral."
            )


        # ====================================================
        # UNEXPECTED CLASS
        # ====================================================

        else:

            st.warning(
                f"Prediction: {prediction.upper()}"
            )


        # ====================================================
        # PROCESSED TEXT
        # ====================================================

        with st.expander("View processed text"):

            st.write(cleaned_text)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">
    Sentify • 3-Class Machine Learning Sentiment Analysis
</div>
""",
    unsafe_allow_html=True
)