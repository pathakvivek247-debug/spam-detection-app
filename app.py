from flask import Flask, render_template, request
import pickle
import pandas as pd
import re

app = Flask(__name__)

# -------------------------------
# Load Model & Vectorizer
# -------------------------------
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[["v1", "v2"]]
df.columns = ["label", "message"]

data = df.values.tolist()

# -------------------------------
# Storage
# -------------------------------
history = []
blocked_messages = set()

# -------------------------------
# Email Detection Fix
# -------------------------------
def is_email(text):
    return re.match(r"[^@]+@[^@]+\.[^@]+", text)

# -------------------------------
# Main Route
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    last_message = None

    if request.method == "POST":

        # -------------------------------
        # BLOCK BUTTON CLICK
        # -------------------------------
        if "block_msg" in request.form:
            msg = request.form["block_msg"]
            blocked_messages.add(msg)

            # Update history status
            for i in range(len(history)):
                if history[i][0] == msg:
                    history[i] = (msg, history[i][1], "Blocked")

            # Reset UI after block
            return render_template(
                "index.html",
                result=None,
                history=history,
                data=data[:20],
                last_message=None
            )

        # -------------------------------
        # NORMAL PREDICTION
        # -------------------------------
        msg = request.form["message"]
        last_message = msg

        if is_email(msg):
            result = "Not Spam"
        else:
            vec = vectorizer.transform([msg])
            pred = model.predict(vec)[0]
            result = "Spam" if pred == 1 else "Not Spam"

        status = "Blocked" if msg in blocked_messages else "Active"
        history.append((msg, result, status))

    return render_template(
        "index.html",
        result=result,
        history=history,
        data=data[:20],
        last_message=last_message
    )

# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)