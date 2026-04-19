print("staring spam detection project...")

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# -------------------------------
# 1. Load Dataset
# -------------------------------
# Make sure spam.csv is in the same folder as this file
df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only required columns
df = df[["v1", "v2"]]
df.columns = ["label", "message"]

# Convert labels: ham -> 0, spam -> 1
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# -------------------------------
# 2. Split Data
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.2, random_state=42
)

# -------------------------------
# 3. Convert Text to Numbers
# -------------------------------
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -------------------------------
# 4. Train Model
# -------------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# -------------------------------
# 5. Evaluate Model
# -------------------------------
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# -------------------------------
# 6. Save Model & Vectorizer
# -------------------------------
with open("spam_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved!")

# -------------------------------
# 7. Prediction Function
# -------------------------------
def predict_message(msg: str) -> str:
    vec = vectorizer.transform([msg])
    pred = model.predict(vec)[0]
    return "Spam" if pred == 1 else "Not Spam"

# -------------------------------
# 8. Try Sample Inputs
# -------------------------------
print(predict_message("Congratulations! You won a lottery"))
print(predict_message("Hey, are we meeting today?"))

print("Project completed successfully!")