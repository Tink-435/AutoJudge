from flask import Flask, request, render_template_string
import joblib
import numpy as np
import re
from scipy.sparse import hstack

app = Flask(__name__)

# Load Trained Artifacts
vectorizer = joblib.load("models/vectorizer.pkl")
scaler = joblib.load("models/scaler.pkl")
regressor = joblib.load("models/regressor.pkl")

# Feature Helpers
SYMBOLS = ['+', '-', '*', '/', '%', '<', '>', '=', '^']
KEYWORDS = [
    'dp', 'dynamic programming', 'graph', 'tree',
    'dfs', 'bfs', 'recursion', 'greedy',
    'bitmask', 'segment tree', 'binary search'
]

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def build_features(text):
    # TF-IDF features
    X_tfidf = vectorizer.transform([text])

    # Numeric features
    num_chars = len(text)
    num_words = len(text.split())
    symbol_count = sum(text.count(sym) for sym in SYMBOLS)

    numeric = np.array([[num_chars, num_words, symbol_count]])
    numeric = scaler.transform(numeric)

    # Keyword features
    keyword_features = [[text.count(k) for k in KEYWORDS]]

    return hstack([X_tfidf, numeric, keyword_features])

# Fixed Threshold Mapping 
def score_to_class(score):
    if score <= 3.0:
        return "Easy"
    elif score <= 5.5:
        return "Medium"
    else:
        return "Hard"

# HTML Template
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AutoJudge</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(to right, #eef2f7, #f8fafc);
            padding: 40px;
        }
        .card {
            max-width: 850px;
            margin: auto;
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.08);
        }
        h1 {
            text-align: center;
            margin-bottom: 5px;
        }
        .subtitle {
            text-align: center;
            color: #555;
            margin-bottom: 25px;
        }
        label {
            font-weight: bold;
            display: block;
            margin-top: 18px;
        }
        input, textarea {
            width: 100%;
            padding: 10px;
            margin-top: 6px;
            border-radius: 6px;
            border: 1px solid #ccc;
            font-size: 14px;
        }
        textarea {
            min-height: 90px;
            resize: vertical;
        }
        button {
            margin-top: 25px;
            padding: 12px 22px;
            font-size: 15px;
            border: none;
            border-radius: 6px;
            background: #2563eb;
            color: white;
            cursor: pointer;
        }
        button:hover {
            background: #1e40af;
        }
        .result {
            margin-top: 30px;
            padding: 18px;
            background: #f1f5ff;
            border-left: 5px solid #2563eb;
            border-radius: 6px;
        }
        .info-link {
            margin-top: 15px;
            color: #2563eb;
            cursor: pointer;
            font-size: 14px;
            text-decoration: underline;
            display: inline-block;
        }
        .info-box {
            display: none;
            margin-top: 15px;
            padding: 14px;
            background: #f9fafb;
            border: 1px dashed #cbd5e1;
            border-radius: 6px;
            font-size: 14px;
            color: #333;
        }
        .footer {
            margin-top: 35px;
            text-align: center;
            font-size: 13px;
            color: #777;
        }
    </style>

    <script>
        function toggleInfo() {
            var box = document.getElementById("infoBox");
            box.style.display = box.style.display === "none" ? "block" : "none";
        }
    </script>
</head>

<body>

<div class="card">
    <h1>AutoJudge</h1>
    <p class="subtitle">
        Predicting programming problem difficulty using textual analysis
    </p>

    <form method="POST">

        <label>Problem Title</label>
        <input type="text" name="title" placeholder="e.g. Longest Path in a Tree" value="{{ title }}">

        <label>Problem Description</label>
        <textarea name="desc" required>{{ desc }}</textarea>

        <label>Input Description</label>
        <textarea name="input_desc">{{ input_desc }}</textarea>

        <label>Output Description</label>
        <textarea name="output_desc">{{ output_desc }}</textarea>

        <button type="submit">Predict Difficulty</button>
    </form>

    {% if result %}
    <div class="result">
        <p><b>Predicted Difficulty Class:</b> {{ result }}</p>
        <p><b>Predicted Difficulty Score:</b> {{ score }}</p>

        <span class="info-link" onclick="toggleInfo()">How does this work?</span>

        <div class="info-box" id="infoBox">
            <p><b>Difficulty Score:</b> Predicted using Random Forest Regressor</p>
            <p><b>Difficulty Class:</b> Derived from the predicted score using fixed thresholds</p>
            <p>
                This design avoids contradictory predictions and ensures
                consistent, interpretable difficulty estimation.
            </p>
        </div>
    </div>
    {% endif %}

    <div class="footer">
        AutoJudge · NLP-based Difficulty Prediction System
    </div>
</div>

</body>
</html>
"""

# Route 
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    score = None

    title = ""
    desc = ""
    input_desc = ""
    output_desc = ""

    if request.method == "POST":
        title = request.form.get("title", "")
        desc = request.form.get("desc", "")
        input_desc = request.form.get("input_desc", "")
        output_desc = request.form.get("output_desc", "")

        combined_text = f"""
        Title: {title}
        Description: {desc}
        Input: {input_desc}
        Output: {output_desc}
        """

        combined_text = clean_text(combined_text)
        X = build_features(combined_text)

        score = round(regressor.predict(X)[0], 2)
        result = score_to_class(score)

    return render_template_string(
        HTML,
        result=result,
        score=score,
        title=title,
        desc=desc,
        input_desc=input_desc,
        output_desc=output_desc
    )

# Run App
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
