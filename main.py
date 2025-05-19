import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# Download necessary NLTK data
nltk.download('stopwords')

# Text Preprocessing Function
def preprocess_text(text):
    ps = PorterStemmer()
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Load and preprocess dataset
def load_and_train_model():
  # data = pd.read_csv("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv", sep='\t', names=['label', 'message'])
    data = pd.read_csv("spam.csv", sep='\t', names=['label', 'message'])
    data['label_num'] = data.label.map({'ham': 0, 'spam': 1})
    data['cleaned'] = data['message'].apply(preprocess_text)

    X_train, X_test, y_train, y_test = train_test_split(data['cleaned'], data['label_num'], test_size=0.2, random_state=42)

    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)
    return model

# Prediction Function
def detect_spam():
    msg = entry.get("1.0", tk.END).strip()
    if not msg:
        messagebox.showwarning("Input Error", "Please enter a message to classify.")
        return
    cleaned_msg = preprocess_text(msg)
    result = model.predict([cleaned_msg])[0]

    if result == 1:
        output = "SPAM ❌"
        result_label.config(text=f"Result: {output}", fg="red")
    else:
        output = "NOT SPAM ✅"
        result_label.config(text=f"Result: {output}", fg="green")


#UI with Tkinter
app = tk.Tk()
app.title("SMS Spam Detection")
app.geometry("600x400")
app.configure(bg="white")


tk.Label(app, text="Enter SMS Text:", font=("Arial", 20), bg="white").pack(pady=25)
entry = tk.Text(app, height=5, width=50, font=("Arial", 14))
entry.pack()

tk.Button(app, text="Check Spam", command=detect_spam, bg="blue", fg="white", font=("Arial", 18)).pack(pady=25)
result_label = tk.Label(app, text="", font=("Arial", 16), bg="white", fg="green")
result_label.pack()

tk.Label(app, text="Created by  ABJD", font=("Arial", 14), bg="white", fg="gray").pack(side=tk.BOTTOM, pady=25)

# Load the model
model = load_and_train_model()

# Run the app
app.mainloop()
