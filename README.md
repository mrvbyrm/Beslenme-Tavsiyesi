# Beslenme-Tavsiyesi

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)]()
[![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white)]()
[![NLP](https://img.shields.io/badge/NLP-spaCy%20%7C%20SentenceTransformers-09A3D5?style=flat-square)]()

Yapay zeka destekli, doğal dil girdisine göre beslenme tavsiyesi sunan bir web uygulaması.

## Proje Hakkında

Kullanıcının serbest metin olarak yazdığı isteği (örn. "öğle yemeği için vegan bir öneri") anlamlandırıp uygun besinleri öneren bir Flask uygulamasıdır. Öneri motoru; `sentence-transformers` ile metinleri vektöre dönüştürüp kosinüs benzerliği üzerinden besin verisiyle eşleştirir, `spaCy` (`PhraseMatcher`) ile de anahtar kelime/anlam eşlemesi yapar.

## Kullanılan Teknolojiler

- Python, Flask
- pandas, scikit-learn
- sentence-transformers (`all-MiniLM-L6-v2`)
- spaCy

## Kurulum ve Çalıştırma

```bash
git clone https://github.com/mrvbyrm/Beslenme-Tavsiyesi.git
cd Beslenme-Tavsiyesi/Beslenme
pip install flask pandas sentence-transformers scikit-learn spacy
python -m spacy download en_core_web_sm
python app.py
```

Uygulama, gerekli veri dosyalarıyla (`food.csv`, `kelime_anlam.csv`) çalışır ve `templates/` altındaki arayüzü sunar.

## İletişim

Merve — [GitHub](https://github.com/mrvbyrm)
