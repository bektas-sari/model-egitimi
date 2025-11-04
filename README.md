# Model Eğitimi (Yapay Zeka + FLASK)

📌 **Türkçe Duygu Analizi - Word2Vec ve Logistic Regression**

Bu proje, Türkçe metinler üzerinde duygu analizi yapmak için Word2Vec gömme yöntemi ve Logistic Regression algoritmasını kullanarak eğitilmiş bir makine öğrenimi modeli sunmaktadır. 
Model, Flask tabanlı modern bir web uygulaması üzerinden kullanılabilir.

---

## 🚀 Projeyi Çalıştırma

Aşağıdaki adımları takip ederek projeyi yerel ortamınızda çalıştırabilirsiniz:

### 1. Gerekli Bağımlılıkları Kurun

```bash
pip install pandas numpy nltk gensim scikit-learn flask joblib
```

### 2. Veri Setini Hazırlama

Proje klasörüne aşağıdaki CSV dosyalarını ekleyin:
- `train.csv`
- `test.csv`

Dosya formatı:

| text                      | label     |
|---------------------------|-----------|
| "Ürün harikaydı!"         | Positive |
| "Hiç beğenmedim."        | Negative |

- "Notr" etiketli veriler analiz dışında tutulur.

### 3. Model Eğitimi

Modeli eğitmek için terminal üzerinden çalıştırın:

```bash
python model_egitimi.py
```

Eğitim sonucu aşağıdaki modeller kaydedilir:

- `word2vec_classifier.pkl`: Logistic Regression modeli
- `word2vec_model.bin`: Word2Vec kelime vektör modeli

### 3. Flask Web Uygulamasını Başlatma

Flask uygulamasını başlatmak için:

```bash
python app.py
```

Ardından tarayıcınızda aşağıdaki URL'yi ziyaret edin:

```
http://127.0.0.1:5000
```

---

## 📁 Proje Yapısı

```
model-egitimi/
├── app.py                  # Flask uygulama dosyası
├── model_egitimi.py        # Model eğitim kodları
├── train.csv               # Eğitim veri seti
├── test.csv                # Test veri seti
├── templates/
│   └── index.html          # Web arayüzü
├── static/
│   ├── style.css           # Stil dosyası
│   └── script.js           # JavaScript dosyası
├── word2vec_classifier.pkl # Eğitilmiş model
├── word2vec_model.bin      # Word2Vec modeli
└── README.md               # Proje dokümantasyonu
```

---

## 🚧 Kullanılan Teknolojiler
- **Python**
- **Flask**
- **NLTK & Gensim**
- **Scikit-learn**
- **HTML, CSS, JavaScript**

---

## 👤 Developer

**Bektaş Sarı**<br>
PhD in Advertising, AI + Creativity researcher<br>
Flutter Developer & Software Educator<br>

- **Email:** [bektas.sari@gmail.com](mailto:bektas.sari@gmail.com)  
- **GitHub:** [github.com/bektas-sari](https://github.com/bektas-sari)  
- **LinkedIn:** [linkedin.com/in/bektas-sari](https://www.linkedin.com/in/bektas-sari)  
- **Researchgate:** [researchgate.net/profile/Bektas-Sari-3](https://www.researchgate.net/profile/Bektas-Sari-3)  
- **Academia:** [independent.academia.edu/bektassari](https://independent.academia.edu/bektassari)
