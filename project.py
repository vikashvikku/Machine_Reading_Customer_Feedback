import streamlit as st
import pandas as pd
import re
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from io import BytesIO

st.set_page_config(
    page_title="Customer Feedback Insights",
    page_icon="📊",  # You can also use an image file like 'logo.png'
    layout="wide"
)
# 🔹 Detect Fastest Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 🔹 Load BERT Model (Cached)
@st.cache_resource
def load_bert_model():
    return pipeline("text-classification", model="priyanshu745/Distilbert", tokenizer="priyanshu745/Distilbert")

sentiment_pipeline = load_bert_model()

# 🔹 Load VADER Sentiment Analyzer
vader = SentimentIntensityAnalyzer()

# 🔹 Text Cleaning Function
def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+|\#', '', text)  # Remove mentions & hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation/numbers
    return text

# 🔹 **Hybrid Sentiment Analysis (BERT + VADER)**
def get_sentiment(text):
    cleaned_text = clean_text(text)

    # **VADER Analysis**
    vader_score = vader.polarity_scores(cleaned_text)['compound']

    # **BERT Analysis**
    with torch.no_grad():
        result = sentiment_pipeline(cleaned_text)[0]

    sentiment_label = result['label']
    sentiment_score = round(result['score'], 4)

    # **Hybrid Decision Logic**
    if sentiment_score < 0.70 or (-0.2 < vader_score < 0.2):
        sentiment_label = "Neutral"
    elif sentiment_label == "POSITIVE":
        sentiment_label = "Positive"
    else:
        sentiment_label = "Negative"

    return sentiment_label, sentiment_score

# 🔹 Load or Process Data
@st.cache_data
def load_or_process_data():
    processed_file = "processed_feedback.csv"

    if os.path.exists(processed_file):
        return pd.read_csv(processed_file)

    df = pd.read_csv("customer_feedback.csv", low_memory=False)
    df['cleaned_feedback'] = df['feedback_text'].apply(clean_text)

    # **Fast Batch Sentiment Processing**
    feedback_list = df['cleaned_feedback'].tolist()
    df['sentiment_label'] = [get_sentiment(feedback)[0] for feedback in feedback_list]

    # 🔹 Save Processed Data
    df.to_csv(processed_file, index=False)
    return df

df = load_or_process_data()

# 🔹 Streamlit UI
st.title("📊 Machine Reading of Customer Feedback")

# 🔹 Sidebar Filters
product_filter = st.sidebar.selectbox("🔍 Select Product", ["All"] + list(df["product_name"].unique()))
location_filter = st.sidebar.selectbox("📍 Select Location", ["All"] + list(df["customer_location"].unique()))
sentiment_filter = st.sidebar.selectbox("😀 Select Sentiment", ["All", "Positive", "Neutral", "Negative"])

filtered_df = df.copy()
if product_filter != "All":
    filtered_df = filtered_df[filtered_df["product_name"] == product_filter]
if location_filter != "All":
    filtered_df = filtered_df[filtered_df["customer_location"] == location_filter]
if sentiment_filter != "All":
    filtered_df = filtered_df[filtered_df["sentiment_label"] == sentiment_filter]

# 🔹 Show Filtered Data
st.write(f"Showing {len(filtered_df)} results")
st.dataframe(filtered_df)

# 🔹 **📈 Fix for Missing Neutral Sentiment in Graph**
st.subheader("📊 Sentiment Distribution")

# **Ensure all categories appear**
sentiment_order = ["Positive", "Neutral", "Negative"]
df['sentiment_label'] = pd.Categorical(df['sentiment_label'], categories=sentiment_order, ordered=True)

# **Fix for missing Neutral category**
sentiment_counts = df['sentiment_label'].value_counts().reindex(sentiment_order, fill_value=0)

# **Plot sentiment distribution**
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='coolwarm', ax=ax)
ax.set_ylabel("Count")
ax.set_xlabel("Sentiment")
ax.set_title("Sentiment Distribution")

# **Ensure Y-axis starts at zero**
ax.set_ylim(0, max(sentiment_counts.values) + 500)

# **Show Exact Counts on Bars**
for i, v in enumerate(sentiment_counts.values):
    ax.text(i, v + 50, str(v), ha='center', fontsize=8, fontweight='bold')

st.pyplot(fig)
# 🔹 📈 **Filtered Sentiment Distribution**
st.subheader("📊 Filtered Sentiment Distribution")
fig2, ax2 = plt.subplots(figsize=(8, 4))
sns.countplot(data=filtered_df, x='sentiment_label', hue='sentiment_label', palette='coolwarm', ax=ax2, order=["Positive", "Neutral", "Negative"], legend=False)
st.pyplot(fig2)
# 🔹 **📝 Real-time Sentiment Analysis**
st.subheader("📝 Live Sentiment Analysis")
user_feedback = st.text_area("Enter customer feedback:")
if user_feedback:
    sentiment, score = get_sentiment(user_feedback)
    st.write(f"**Sentiment:** {sentiment} (Score: {score}) ✅ (Hybrid Model Applied!)")

# 🔹 Download Processed Data
st.subheader("📥 Download Processed Data")
def convert_df_to_csv(df):
    output = BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return output

csv_file = convert_df_to_csv(df)
st.download_button(label="Download CSV", data=csv_file, file_name="processed_feedback.csv", mime="text/csv")

st.success("✅ Machine Reading of Customer Feedback Successfully Integrated!")
