from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

app = Flask(__name__)

# Model ve vektörleyiciyi yükle
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Metin ön işleme fonksiyonu
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('turkish'))

def preprocess_text(text):
    words = nltk.word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(words)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form['text']
        processed_text = preprocess_text(text)
        vectorized_text = vectorizer.transform([processed_text])
        prediction = model.predict(vectorized_text)
        confidence = model.predict_proba(vectorized_text).max()
        
        result = {
            'text': text,
            'prediction': 'Olumlu' if prediction[0] == 1 else 'Olumsuz',
            'confidence': f"{confidence:.2%}"
        }
        return render_template('index.html', result=result)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
