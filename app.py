import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")  # Enable full-width layout

st.title("Twitter Mental Health & Sentiment Analysis Dashboard")

# Load the CSV
df = pd.read_csv("/Mental_Health_X/Mental_Health_Twitter/data/tweets_sentiments_dataset.csv")  # Adjust path as needed

# Dataset Overview Section
st.subheader("📄 Dataset Overview")
st.markdown("---")
if st.checkbox("🔍 Show Raw Data :"):
    st.dataframe(df, use_container_width=True)

st.markdown("### Basic Statistics")
col_meta1, col_meta2= st.columns(2)

with col_meta2:
    st.write("**Shape of Dataset :**")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
with col_meta1:
    st.write("**Mental Health Distribution :**")
    st.write(df['mental_health'].value_counts())

# Full-width for describe table to prevent clipping
st.write("**Summary Statistics**")
st.dataframe(df.describe(), use_container_width=True)

# Charts Section
st.markdown("## 📊 Visualizations")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Mental Health Classification")
    mh_counts = df['mental_health'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(mh_counts, labels=mh_counts.index, autopct='%1.1f%%')
    st.pyplot(fig1)


with col2:
    if 'sentiment' in df.columns:
        st.subheader("Sentiment Distribution")
        fig2, ax2 = plt.subplots()
        sns.histplot(x=df['sentiment'], hue=df['sentiment'], multiple='stack', ax=ax2)
        st.pyplot(fig2)
    else:
        st.warning("No 'sentiment' column found in dataset.")


st.header('Conclusion')
st.markdown("---")
st.write('In this project, we conducted a comprehensive sentiment and mental health analysis on Twitter posts using an combination of Kaggle-sourced dataset and Twitter(X) API. After scraping, integrating, cleaning and preprocessing the data, we applied a sentiment analysis on the cleened dataset. The dataset is classified into "Negative" , "Neutral" , "Positive" text. Further rule-based and keyword-enhanced classification approach is used on negatively classified tweets to distinguish between "depressed " and " undepressed " tweets.')
st.write("Out of a total of 20,100 tweets analyzed:")
st.write("19,955 tweets (≈ 99.28%) were classified as undepressed")
st.write("145 tweets (≈ 0.72%) were identified as depressed")
st.subheader("Future Work")

future_points = [
    "**Integrate advanced NLP models** such as BERT or DistilRoBERTa fine-tuned for mental health classification.",
    "**Build a feedback loop** for human moderation and validation to improve accuracy over time.",
    "**Incorporate time-series analysis** to track emotional patterns across user timelines.",
    "**Collaborate with mental health professionals** to validate model decisions and ensure ethical application.",
    "**Enhance privacy safeguards**, anonymize data, and consider opt-in detection frameworks.",
    "**Expand data sources** beyond Twitter — like Reddit or forums — to get richer, more nuanced mental health signals."
]

for point in future_points:
    st.markdown(f"- {point}")