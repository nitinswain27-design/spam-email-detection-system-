
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Step 1: Dataset (sample)
# -----------------------------
data = {
    "text": [
        "Win a free iPhone now",
        "Meeting scheduled at 10 AM",
        "Claim your free prize today",
        "Project discussion tomorrow",
        "Congratulations you won a lottery",
        "Please submit the assignment",
        "Limited offer click now",
        "Team meeting has been postponed",
        "Free cashback waiting for you",
        "Let us complete the project"
    ],
    "label": ["spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham"]
}

df = pd.DataFrame(data)

# -----------------------------
# Step 2: Data Preprocessing
# -----------------------------
df["label"] = df["label"].map({"spam": 1, "ham": 0})

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# -----------------------------
# Step 3: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Step 4: Model Training
# -----------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

# -----------------------------
# Step 5: Model Evaluation
# -----------------------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# Step 6: Prediction Function
# -----------------------------
def predict_email(message):
    message_vector = vectorizer.transform([message])
    result = model.predict(message_vector)
    return "Spam" if result[0] == 1 else "Not Spam"

# -----------------------------
# Step 7: Test Prediction
# -----------------------------
test_email = "Free reward waiting for you"
print("\nEmail:", test_email)
print("Prediction:", predict_email(test_email))
