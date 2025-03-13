import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# NLTK paketlerini indir
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('turkish'))

# 1. Veri Setlerini Yükle
train_path = "train.csv"
test_path = "test.csv"
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# 2. Etiketleri dönüştür (Positive: 1, Negative: 0, Notr kaldırıldı)
label_mapping = {"Positive": 1, "Negative": 0}
train_df = train_df[train_df['label'].isin(label_mapping.keys())]
test_df = test_df[test_df['label'].isin(label_mapping.keys())]
train_df['label'] = train_df['label'].map(label_mapping)
test_df['label'] = test_df['label'].map(label_mapping)

# 2. Metin Ön İşleme

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return tokens

train_df['tokens'] = train_df['text'].apply(preprocess_text)
test_df['tokens'] = test_df['text'].apply(preprocess_text)

# 2. Word2Vec Modeli Eğitimi
from gensim.models import Word2Vec

sentences = train_df['tokens'].tolist()
w2v_model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=2, workers=4)
w2v_model.train(sentences, total_examples=len(sentences), epochs=10)

# 3. Metinleri Vektörleştir
def vectorize_text(tokens, model):
    vectors = [model.wv[t] for t in tokens if t in model.wv]
    return np.mean(vectors, axis=0) if len(vectors) > 0 else np.zeros(model.vector_size)

train_df['vector'] = train_df['tokens'].apply(lambda x: vectorize_text(x, w2v_model))
test_df['vector'] = test_df['tokens'].apply(lambda x: vectorize_text(x, w2v_model))

# 4. Model Eğitimi
from sklearn.linear_model import LogisticRegression

X_train = np.vstack(train_df['vector'].values)
y_train = train_df['label'].values
X_test = np.vstack(test_df['vector'].values)
y_test = test_df['label'].values

clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# 5. Modeli Kaydet
joblib.dump(clf, 'word2vec_classifier.pkl')
w2v_model.save('word2vec_model.bin')

print("Model başarıyla eğitildi ve kaydedildi!")