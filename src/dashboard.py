import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import hstack

from preprocessing import preprocess_text


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Emotion AI | Multi-Emotion Detection",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# FIX: Streamlit's Markdown parser turns any HTML block into a
# plain-text code block as soon as it hits a blank line followed
# by indented content (this is what was causing raw HTML/CSS to
# show up as visible text on the page instead of being rendered).
#
# This patch wraps st.markdown / st.sidebar.markdown so that any
# call using unsafe_allow_html=True has its blank lines removed
# and each line's leading indentation stripped before rendering.
# That keeps the HTML as a single unbroken block, so the parser
# always treats it as raw HTML passthrough — regardless of how
# the triple-quoted string is indented in the source.
# ==========================================================

def _clean_html(body: str) -> str:
    lines = [ln.strip() for ln in body.strip().splitlines()]
    lines = [ln for ln in lines if ln]
    return " ".join(lines)


def _make_patched_markdown(original_markdown):
    def _patched(body="", *args, **kwargs):
        if kwargs.get("unsafe_allow_html") and isinstance(body, str) and "\n" in body:
            body = _clean_html(body)
        return original_markdown(body, *args, **kwargs)
    return _patched


st.markdown = _make_patched_markdown(st.markdown)
st.sidebar.markdown = _make_patched_markdown(st.sidebar.markdown)


# ==========================================================
# PREMIUM FRONTEND CSS
# ==========================================================

st.markdown("""
<style>

/* ==========================================================
   GLOBAL
   ========================================================== */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 5% 5%,
            rgba(37, 99, 235, 0.12),
            transparent 25%
        ),
        radial-gradient(
            circle at 95% 10%,
            rgba(6, 182, 212, 0.08),
            transparent 25%
        ),
        #07111f;

    color: #e5eef8;
}


/* ==========================================================
   REMOVE DEFAULT STREAMLIT ELEMENTS
   ========================================================== */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}


/* ==========================================================
   MAIN CONTAINER
   ========================================================== */

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* ==========================================================
   SIDEBAR
   ========================================================== */

section[data-testid="stSidebar"] {
    background: #091522;
    border-right: 1px solid rgba(148, 163, 184, 0.12);
}

section[data-testid="stSidebar"] * {
    color: #dbeafe;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(148, 163, 184, 0.12);
}


/* ==========================================================
   HEADER
   ========================================================== */

.ai-header {
    position: relative;

    padding: 28px 32px;

    border-radius: 24px;

    background:
        linear-gradient(
            135deg,
            rgba(15, 35, 58, 0.97),
            rgba(8, 22, 37, 0.97)
        );

    border: 1px solid rgba(96, 165, 250, 0.18);

    box-shadow:
        0 20px 50px rgba(0, 0, 0, 0.25);

    margin-bottom: 32px;
}

.ai-brand {
    display: flex;
    align-items: center;
    gap: 15px;
}

.ai-logo {
    width: 52px;
    height: 52px;

    border-radius: 15px;

    display: flex;
    align-items: center;
    justify-content: center;

    background:
        linear-gradient(
            135deg,
            #2563eb,
            #0891b2
        );

    font-size: 25px;

    box-shadow:
        0 8px 25px rgba(37, 99, 235, 0.35);
}

.ai-title {
    font-size: 29px;
    font-weight: 800;
    letter-spacing: -1px;
    color: #f8fafc;
}

.ai-subtitle {
    font-size: 13px;
    color: #94a3b8;
    margin-top: 4px;
}

.status-pill {
    position: absolute;
    right: 28px;
    top: 30px;

    padding: 8px 14px;

    border-radius: 999px;

    background: rgba(34, 197, 94, 0.08);

    border: 1px solid rgba(34, 197, 94, 0.22);

    color: #86efac;

    font-size: 11px;
    font-weight: 700;

    letter-spacing: 0.5px;
}

.status-dot {
    display: inline-block;

    width: 7px;
    height: 7px;

    background: #22c55e;

    border-radius: 50%;

    margin-right: 7px;

    box-shadow:
        0 0 10px #22c55e;
}


/* ==========================================================
   SECTION HEADINGS
   ========================================================== */

.section-label {
    font-size: 10px;

    font-weight: 700;

    letter-spacing: 1.6px;

    color: #60a5fa;

    margin-bottom: 6px;

    text-transform: uppercase;
}

.section-title {
    font-size: 23px;

    font-weight: 750;

    letter-spacing: -0.4px;

    color: #f8fafc;

    margin-bottom: 5px;
}

.section-description {
    color: #94a3b8;

    font-size: 13px;

    margin-bottom: 18px;
}


/* ==========================================================
   INPUT CARD
   ========================================================== */

.input-card {
    background:
        linear-gradient(
            145deg,
            rgba(15, 30, 48, 0.94),
            rgba(9, 22, 37, 0.94)
        );

    border: 1px solid rgba(148, 163, 184, 0.13);

    border-radius: 22px;

    padding: 23px;

    margin-bottom: 16px;

    box-shadow:
        0 15px 40px rgba(0, 0, 0, 0.18);
}


/* ==========================================================
   TEXT AREA
   ========================================================== */

textarea {
    background: #0a1727 !important;

    color: #f8fafc !important;

    border: 1px solid rgba(96, 165, 250, 0.18) !important;

    border-radius: 15px !important;

    font-size: 15px !important;

    line-height: 1.6 !important;
}

textarea::placeholder {
    color: #64748b !important;
}

textarea:focus {
    border: 1px solid #3b82f6 !important;

    box-shadow:
        0 0 0 2px rgba(59, 130, 246, 0.12) !important;
}


/* ==========================================================
   ANALYZE BUTTON
   ========================================================== */

.stButton > button {
    width: 100%;

    height: 50px;

    border-radius: 14px;

    border: 1px solid rgba(96, 165, 250, 0.25);

    background:
        linear-gradient(
            135deg,
            #2563eb,
            #0891b2
        );

    color: white;

    font-size: 13px;

    font-weight: 700;

    letter-spacing: 0.4px;

    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);

    box-shadow:
        0 12px 28px rgba(37, 99, 235, 0.30);

    border-color: rgba(125, 211, 252, 0.45);
}


/* ==========================================================
   RESULT CARDS
   ========================================================== */

.result-card {
    min-height: 185px;

    border-radius: 22px;

    padding: 25px;

    background:
        linear-gradient(
            145deg,
            rgba(15, 30, 48, 0.97),
            rgba(9, 22, 37, 0.97)
        );

    border: 1px solid rgba(148, 163, 184, 0.13);

    box-shadow:
        0 15px 40px rgba(0, 0, 0, 0.20);
}

.result-label {
    font-size: 10px;

    font-weight: 700;

    letter-spacing: 1.4px;

    color: #64748b;

    text-transform: uppercase;
}

.result-emotion {
    font-size: 34px;

    font-weight: 800;

    margin-top: 25px;

    letter-spacing: -0.7px;
}

.result-icon {
    font-size: 36px;

    margin-right: 8px;
}

.confidence-number {
    font-size: 38px;

    font-weight: 800;

    color: #f8fafc;

    margin-top: 21px;

    letter-spacing: -1px;
}

.confidence-caption {
    color: #64748b;

    font-size: 11px;

    margin-top: 5px;
}


/* ==========================================================
   PROGRESS BAR
   ========================================================== */

div[data-testid="stProgress"] {
    margin-top: -7px;
}

div[data-testid="stProgress"] > div {
    background: #13263b;
}

div[data-testid="stProgress"] > div > div {
    background:
        linear-gradient(
            90deg,
            #2563eb,
            #06b6d4
        );
}


/* ==========================================================
   INFORMATION CARD
   ========================================================== */

.info-card {
    background: #0b1727;

    border: 1px solid rgba(148, 163, 184, 0.10);

    border-radius: 18px;

    padding: 20px;
}

.info-title {
    color: #f8fafc;

    font-weight: 700;

    font-size: 14px;

    margin-bottom: 10px;
}

.info-text {
    color: #94a3b8;

    font-size: 12px;

    line-height: 1.7;
}


/* ==========================================================
   METRIC CARDS
   ========================================================== */

.metric-card {
    background:
        linear-gradient(
            145deg,
            rgba(15, 30, 48, 0.90),
            rgba(10, 23, 38, 0.90)
        );

    border: 1px solid rgba(148, 163, 184, 0.11);

    border-radius: 18px;

    padding: 19px;

    min-height: 105px;

    transition:
        transform 0.2s ease,
        border-color 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-2px);

    border-color:
        rgba(96, 165, 250, 0.28);
}

.metric-value {
    font-size: 24px;

    font-weight: 800;

    color: #f8fafc;

    letter-spacing: -0.5px;
}

.metric-name {
    color: #64748b;

    font-size: 10px;

    font-weight: 600;

    margin-top: 5px;

    text-transform: uppercase;

    letter-spacing: 0.8px;
}


/* ==========================================================
   PROCESSED TEXT CARD
   ========================================================== */

.processed-box {
    background: #0a1727;

    border: 1px solid rgba(148, 163, 184, 0.10);

    border-radius: 16px;

    padding: 17px;

    color: #94a3b8;

    font-size: 13px;

    line-height: 1.7;

    min-height: 80px;
}


/* ==========================================================
   DATAFRAME
   ========================================================== */

div[data-testid="stDataFrame"] {
    border-radius: 15px;

    overflow: hidden;

    border: 1px solid rgba(148, 163, 184, 0.12);
}


/* ==========================================================
   DOWNLOAD BUTTON
   ========================================================== */

.stDownloadButton > button {
    border-radius: 12px;

    background: #0d1d30;

    color: #cbd5e1;

    border: 1px solid rgba(148, 163, 184, 0.16);

    font-weight: 600;

    transition: all 0.2s ease;
}

.stDownloadButton > button:hover {
    border-color: rgba(96, 165, 250, 0.35);

    color: white;
}


/* ==========================================================
   DIVIDERS
   ========================================================== */

hr {
    border-color:
        rgba(148, 163, 184, 0.10) !important;
}


/* ==========================================================
   FOOTER
   ========================================================== */

.ai-footer {
    text-align: center;

    padding: 35px 10px 10px;

    color: #64748b;

    font-size: 11px;

    line-height: 1.8;
}

.ai-footer strong {
    color: #94a3b8;
}


/* ==========================================================
   ALERTS
   ========================================================== */

div[data-testid="stAlert"] {
    border-radius: 14px;
}


/* ==========================================================
   MOBILE
   ========================================================== */

@media (max-width: 768px) {

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .ai-header {
        padding: 22px;
    }

    .ai-title {
        font-size: 23px;
    }

    .status-pill {
        display: none;
    }

    .result-emotion {
        font-size: 28px;
    }

    .confidence-number {
        font-size: 32px;
    }
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# LOAD FINAL EXPERIMENT 22 MODEL
# ==========================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "experiment22_emotion_model.pkl"
    )

    word_vectorizer = joblib.load(
        "word_tfidf_vectorizer.pkl"
    )

    char_vectorizer = joblib.load(
        "char_tfidf_vectorizer.pkl"
    )

    char_weight = joblib.load(
        "char_weight.pkl"
    )

    raw_char_vectorizer = joblib.load(
        "raw_char_tfidf_vectorizer.pkl"
    )

    raw_char_weight = joblib.load(
        "raw_char_weight.pkl"
    )

    raw_word_vectorizer = joblib.load(
        "experiment22_raw_word_vectorizer.pkl"
    )

    raw_word_weight = joblib.load(
        "experiment22_raw_word_weight.pkl"
    )

    return (
        model,
        word_vectorizer,
        char_vectorizer,
        char_weight,
        raw_char_vectorizer,
        raw_char_weight,
        raw_word_vectorizer,
        raw_word_weight
    )


(
    model,
    word_vectorizer,
    char_vectorizer,
    char_weight,
    raw_char_vectorizer,
    raw_char_weight,
    raw_word_vectorizer,
    raw_word_weight
) = load_model()


# ==========================================================
# EMOTION LABELS
# ==========================================================

emotion_names = [
    "Sadness",
    "Joy",
    "Love",
    "Anger",
    "Fear",
    "Surprise"
]


# ==========================================================
# EMOTION EMOJIS
# ==========================================================

emoji = {
    "Sadness": "😢",
    "Joy": "😄",
    "Love": "❤️",
    "Anger": "😠",
    "Fear": "😨",
    "Surprise": "😲"
}


# ==========================================================
# EMOTION COLORS
# ==========================================================

colors = {
    "Sadness": "#60A5FA",
    "Joy": "#34D399",
    "Love": "#F472B6",
    "Anger": "#FB7185",
    "Fear": "#FBBF24",
    "Surprise": "#A78BFA"
}


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.markdown("""
<div style="
    font-size:25px;
    font-weight:800;
    color:#f8fafc;
    margin-bottom:4px;
">
    Emotion AI
</div>

<div style="
    font-size:11px;
    color:#64748b;
    margin-bottom:24px;
    letter-spacing:0.4px;
">
    MULTI-EMOTION INTELLIGENCE
</div>
""", unsafe_allow_html=True)


st.sidebar.markdown("""
<div style="
    background:#0b1727;
    border:1px solid rgba(148,163,184,0.10);
    border-radius:16px;
    padding:17px;
">
    
<div style="
    font-size:11px;
    font-weight:700;
    letter-spacing:1px;
    color:#60a5fa;
    margin-bottom:10px;
">
MODEL
</div>

<div style="
    font-size:15px;
    font-weight:700;
    color:#f8fafc;
    margin-bottom:8px;
">
Complement Naïve Bayes
</div>

<div style="
    font-size:11px;
    color:#64748b;
    line-height:1.6;
">
TF-IDF Feature Fusion<br>
Experiment 22 Architecture
</div>

</div>
""", unsafe_allow_html=True)


st.sidebar.markdown("<br>", unsafe_allow_html=True)


st.sidebar.markdown("""
<div style="
    font-size:11px;
    font-weight:700;
    letter-spacing:1.2px;
    color:#60a5fa;
    margin-bottom:10px;
">
SYSTEM INFORMATION
</div>
""", unsafe_allow_html=True)


st.sidebar.markdown("""
<div style="
    font-size:12px;
    color:#94a3b8;
    line-height:2;
">

<b style="color:#cbd5e1;">Dataset</b><br>
Dair-AI Emotion

<br>

<b style="color:#cbd5e1;">Classes</b><br>
6 emotions

<br>

<b style="color:#cbd5e1;">Accuracy</b><br>
<span style="color:#34d399;font-weight:700;">90.40%</span>

</div>
""", unsafe_allow_html=True)


st.sidebar.markdown("---")


st.sidebar.markdown("""
<div style="
    font-size:11px;
    font-weight:700;
    letter-spacing:1.2px;
    color:#60a5fa;
    margin-bottom:12px;
">
TRY AN EXAMPLE
</div>
""", unsafe_allow_html=True)


examples = [
    ("😊", "I got promoted today."),
    ("😢", "Nobody understands me."),
    ("😠", "I hate this weather."),
    ("😨", "I'm scared of the dark."),
    ("❤️", "I love my parents."),
    ("😲", "Wow! That's amazing.")
]

for icon, example in examples:

    st.sidebar.markdown(
        f"""
        <div style="
            padding:8px 0;
            color:#94a3b8;
            font-size:11px;
            border-bottom:1px solid rgba(148,163,184,0.06);
        ">
        <span style="font-size:14px;">{icon}</span>
        &nbsp; {example}
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================================
# SESSION HISTORY
# ==========================================================

if "history" not in st.session_state:
    st.session_state.history = []


# ==========================================================
# PREMIUM HEADER
# ==========================================================

st.markdown("""
<div class="ai-header">

    <div class="status-pill">
        <span class="status-dot"></span>
        MODEL ONLINE
    </div>

    <div class="ai-brand">

        <div class="ai-logo">
            ✦
        </div>

        <div>

            <div class="ai-title">
                Emotion AI
            </div>

            <div class="ai-subtitle">
                Multi-Emotion Intelligence & Text Analysis Platform
            </div>

        </div>

    </div>

</div>
""", unsafe_allow_html=True)


# ==========================================================
# TEXT ANALYSIS SECTION
# ==========================================================

st.markdown("""
<div class="section-label">
    TEXT ANALYSIS
</div>

<div class="section-title">
    Analyze emotional context
</div>

<div class="section-description">
    Enter a sentence and let the machine-learning model identify
    the dominant emotion.
</div>
""", unsafe_allow_html=True)


st.markdown(
    '<div class="input-card">',
    unsafe_allow_html=True
)


text = st.text_area(
    "Your text",
    height=145,
    placeholder="Example: I am feeling very happy today!",
    label_visibility="collapsed"
)


st.markdown(
    '</div>',
    unsafe_allow_html=True
)


predict_clicked = st.button(
    "✦  ANALYZE EMOTION",
    width="stretch"
)


# ==========================================================
# PREDICTION
# ==========================================================

if predict_clicked:

    if text.strip() == "":

        st.warning(
            "Please enter a sentence."
        )

        st.stop()


    # ======================================================
    # PREPROCESS
    # ======================================================

    processed = preprocess_text(
        text
    )


    # ======================================================
    # FEATURE EXTRACTION
    # ======================================================

    # Processed Word TF-IDF
    X_word = word_vectorizer.transform(
        [processed]
    )


    # Processed Character TF-IDF
    X_char = char_vectorizer.transform(
        [processed]
    )

    X_char = X_char.multiply(
        char_weight
    )


    # Raw Character TF-IDF
    X_raw_char = raw_char_vectorizer.transform(
        [text]
    )

    X_raw_char = X_raw_char.multiply(
        raw_char_weight
    )


    # Raw Word TF-IDF
    X_raw_word = raw_word_vectorizer.transform(
        [text]
    )

    X_raw_word = X_raw_word.multiply(
        raw_word_weight
    )


    # ======================================================
    # FEATURE FUSION
    # ======================================================

    X = hstack(
        [
            X_word,
            X_char,
            X_raw_char,
            X_raw_word
        ],
        format="csr"
    )


    # ======================================================
    # PREDICTION
    # ======================================================

    probabilities = model.predict_proba(
        X
    )[0]


    prediction = model.predict(
        X
    )[0]


    # ======================================================
    # DETERMINE EMOTION
    # ======================================================

    if isinstance(
        prediction,
        str
    ):

        emotion = prediction.title()

        if emotion not in emotion_names:

            idx = int(
                np.argmax(
                    probabilities
                )
            )

            emotion = emotion_names[idx]

        else:

            idx = emotion_names.index(
                emotion
            )

    else:

        idx = int(
            prediction
        )

        emotion = emotion_names[idx]


    confidence = (
        probabilities[idx] * 100
    )


    # ======================================================
    # RESULT HEADER
    # ======================================================

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-label">
        ANALYSIS RESULT
    </div>

    <div class="section-title">
        Detected emotion
    </div>
    """, unsafe_allow_html=True)


    # ======================================================
    # RESULT CARDS
    # ======================================================

    result_col, confidence_col = st.columns(
        2,
        gap="medium"
    )


    with result_col:

        st.markdown(
            f"""
            <div class="result-card">

                <div class="result-label">
                    Dominant emotion
                </div>

                <div
                    class="result-emotion"
                    style="color:{colors[emotion]};"
                >

                    <span class="result-icon">
                        {emoji[emotion]}
                    </span>

                    {emotion}

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with confidence_col:

        st.markdown(
            f"""
            <div class="result-card">

                <div class="result-label">
                    Prediction confidence
                </div>

                <div class="confidence-number">
                    {confidence:.2f}%
                </div>

                <div class="confidence-caption">
                    Estimated confidence for the detected emotion
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(
            float(confidence) / 100
        )


    # ======================================================
    # MODEL STATISTICS
    # ======================================================

    st.markdown("<br>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(
        4,
        gap="medium"
    )


    with m1:

        st.markdown("""
        <div class="metric-card">

            <div class="metric-value">
                90.40%
            </div>

            <div class="metric-name">
                Experimental Test Accuracy
            </div>

        </div>
        """, unsafe_allow_html=True)


    with m2:

        st.markdown("""
        <div class="metric-card">

            <div class="metric-value">
                6
            </div>

            <div class="metric-name">
                Emotion Classes
            </div>

        </div>
        """, unsafe_allow_html=True)


    with m3:

        st.markdown("""
        <div class="metric-card">

            <div class="metric-value">
                4
            </div>

            <div class="metric-name">
                Feature Branches
            </div>

        </div>
        """, unsafe_allow_html=True)


    with m4:

        st.markdown("""
        <div class="metric-card">

            <div class="metric-value">
                CNB
            </div>

            <div class="metric-name">
                Classifier
            </div>

        </div>
        """, unsafe_allow_html=True)


    # ======================================================
    # PROCESSED TEXT / SUMMARY
    # ======================================================

    st.markdown("<br>", unsafe_allow_html=True)

    text_col, summary_col = st.columns(
        2,
        gap="medium"
    )


    with text_col:

        st.markdown("""
        <div class="section-label">
            NLP PROCESSING
        </div>

        <div class="section-title">
            Processed text
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="processed-box">
                {processed}
            </div>
            """,
            unsafe_allow_html=True
        )


    with summary_col:

        st.markdown("""
        <div class="section-label">
            PREDICTION SUMMARY
        </div>

        <div class="section-title">
            Classification details
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="info-card">

                <div class="info-title">
                    {emoji[emotion]} {emotion}
                </div>

                <div class="info-text">
                    The fused TF-IDF representation was evaluated
                    using the trained Complement Naïve Bayes classifier.
                    The highest predicted probability corresponds to
                    <b style="color:{colors[emotion]};">
                        {emotion}
                    </b>
                    with a confidence of
                    <b style="color:#f8fafc;">
                        {confidence:.2f}%
                    </b>.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ======================================================
    # PROBABILITY DISTRIBUTION
    # ======================================================

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-label">
        PROBABILITY ANALYSIS
    </div>

    <div class="section-title">
        Emotion probability distribution
    </div>

    <div class="section-description">
        Probability assigned by the classifier to each emotion class.
    </div>
    """, unsafe_allow_html=True)


    # ======================================================
    # PROBABILITY BARS
    # ======================================================

    for emotion_name, probability in zip(
        emotion_names,
        probabilities
    ):

        percentage = probability * 100

        st.markdown(
            f"""
            <div style="
                margin-bottom:15px;
            ">

                <div style="
                    display:flex;
                    justify-content:space-between;
                    margin-bottom:6px;
                ">

                    <span style="
                        color:#cbd5e1;
                        font-size:12px;
                        font-weight:600;
                    ">
                        {emoji[emotion_name]}
                        &nbsp; {emotion_name}
                    </span>

                    <span style="
                        color:#94a3b8;
                        font-size:12px;
                        font-weight:600;
                    ">
                        {percentage:.2f}%
                    </span>

                </div>

                <div style="
                    width:100%;
                    height:8px;
                    background:#122338;
                    border-radius:999px;
                    overflow:hidden;
                ">

                    <div style="
                        width:{percentage}%;
                        height:100%;
                        background:{colors[emotion_name]};
                        border-radius:999px;
                    ">
                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ======================================================
    # PROBABILITY TABLE
    # ======================================================

    st.markdown("<br>", unsafe_allow_html=True)

    probability_df = pd.DataFrame(
        {
            "Emotion": emotion_names,
            "Probability (%)": [
                round(
                    p * 100,
                    2
                )
                for p in probabilities
            ]
        }
    )


    with st.expander(
        "View detailed probability table"
    ):

        st.dataframe(
            probability_df,
            width="stretch",
            hide_index=True
        )


    # ======================================================
    # SAVE HISTORY
    # ======================================================

    st.session_state.history.append(
        {
            "Sentence": text,
            "Emotion": emotion,
            "Confidence (%)": round(
                confidence,
                2
            )
        }
    )


# ==========================================================
# PREDICTION HISTORY
# ==========================================================

if len(
    st.session_state.history
) > 0:

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-label">
        ACTIVITY
    </div>

    <div class="section-title">
        Prediction history
    </div>

    <div class="section-description">
        Previous emotion predictions from this session.
    </div>
    """, unsafe_allow_html=True)


    df = pd.DataFrame(
        st.session_state.history
    )


    st.dataframe(
        df,
        width="stretch",
        hide_index=True
    )


    csv = df.to_csv(
        index=False
    ).encode(
        "utf-8"
    )


    st.download_button(
        "↓  Download Prediction History",
        csv,
        "emotion_prediction_history.csv",
        "text/csv"
    )


# ==========================================================
# FOOTER
# ==========================================================

st.markdown("""
<div class="ai-footer">

    <div style="
        font-size:16px;
        margin-bottom:6px;
        color:#cbd5e1;
        font-weight:700;
    ">
        ✦ Emotion AI
    </div>

    Multi-Emotion Detection System

    <br>

    Python · Streamlit · TF-IDF · Complement Naïve Bayes

    <br><br>

    <strong>
        Experimental test accuracy: 90.40%
    </strong>

</div>
""", unsafe_allow_html=True)