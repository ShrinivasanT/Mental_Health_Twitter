import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.title("🧠 Mental Health & Sentiment Analysis Dashboard")

# Load the CSV
df = pd.read_csv("data/tweets_sentiments_dataset.csv")
print(df.head())

# Mental health pie chart
st.subheader("Mental Health Classification")
mh_counts = df['mental_health'].value_counts()
fig1, ax1 = plt.subplots()
ax1.pie(mh_counts, labels=mh_counts.index, autopct='%1.1f%%')
st.pyplot(fig1)

# Sentiment chart (if column exists)
if 'sentiment' in df.columns:
    st.subheader("Sentiment Distribution")
    sent_counts = df['sentiment'].value_counts()
    fig2, ax2 = plt.subplots()
    ax2.bar(sent_counts.index, sent_counts.values, color='skyblue')
    st.pyplot(fig2)

# Show raw data
if st.checkbox("Show raw data"):
    st.dataframe(df)
