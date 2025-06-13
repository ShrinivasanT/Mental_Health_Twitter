# Mental Health & Sentiment Analyzer using Social Media Data

This project analyzes Twitter data to detect sentiment and potential signs of mental health distress using machine learning and NLP techniques. It aims to highlight patterns in emotional expression and raise awareness of mental health in online communities.

![Dashboard]([https://your-screenshot-link-if-any](https://mentalhealthtwitter.streamlit.app/))

---

## 📌 Features

- ✅ Preprocess and clean tweet datasets
- ✅ Sentiment analysis (Positive / Negative / Neutral)
- ✅ Mental health classification (Depressed / Undepressed)
- ✅ Filters out news or misleading headlines
- ✅ Visual dashboard with Streamlit: Pie charts, Histograms, and Stats
- ✅ Fully interactive and user-friendly UI

---

## 📂 Dataset Used

1. [`Mental-Health-Twitter`](https://www.kaggle.com/datasets)  
2. `tweets_pulled.csv` (custom)

The combined dataset was preprocessed, cleaned (removing mentions, hashtags, links), and saved as `tweets_sentiments_dataset.csv`.

---

## ⚙️ Tech Stack

- **Python**
- **Pandas**, **Matplotlib**, **Seaborn**
- **Vader** for Sentiment analysis
- **Streamlit** for frontend visualization

---

## 🧪 Model & Classification Logic

- **Sentiment Analysis:** Vader based sentiment analysis
- **Mental Health Classifier:** Rule-based keyword filtering
- **News Detection:** Avoids false positives from headlines by filtering known news phrases

---

## 📊 Dashboard Preview

- **Mental Health Distribution** (Pie Chart)
- **Sentiment Distribution** (Histogram)
- **Dataset Overview**: `shape`, `describe()`, `value_counts()`
- **Raw Data View** toggle

---

## 🚀 How to Run

### 📦 Setup

```bash
git clone https://github.com/ShrinivasanT/Mental_Health_Twitter.git
cd Mental_Health_Twitter
pip install -r requirements.txt
