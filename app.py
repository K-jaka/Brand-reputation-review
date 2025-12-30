import streamlit as st
import pandas as pd
from transformers import pipeline
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

# 1. Page Configuration & Title
st.set_page_config(page_title="Brand Reputation for the year 2023", layout="wide")
st.title("2023 Brand Reputation Monitor")

# 2. AI Implementation (Hugging Face)
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_analyzer = load_sentiment_model()

# 3. Data Loading
@st.cache_data
def load_local_data():
    df = pd.read_csv("scraped_reviews.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

# Helper function to clean and capitalize product names
def clean_product_name(name):
    if not isinstance(name, str):
        return name
    
    # 1. Replace dashes with spaces
    name = name.replace('-', ' ')
    
    # 2. Remove the first word
    parts = name.split()
    if len(parts) > 1:
        name = " ".join(parts[1:])
    
    # 3. Remove the last number if the name ends with one
    name = re.sub(r'\s\d+$', '', name).strip()
    
    # 4. Capitalize the first word
    if name:
        name = name[0].upper() + name[1:]
        
    return name

try:
    df = load_local_data()
except FileNotFoundError:
    st.error("'scraped_reviews.csv' not found. Please run your scraper first!")
    st.stop()

# 4. Sidebar Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to:", ["Products", "Testimonials", "Reviews - with sentiment"])

# Section: Products (Page 1)
if section == "Products":
    st.header("Product list")
    
    product_df = df[['title']].drop_duplicates().copy()
    product_df['title'] = product_df['title'].apply(clean_product_name)
    
    product_df = product_df.rename(columns={'title': 'Product names'})
    product_df.index = range(1, len(product_df) + 1)
    
    st.dataframe(product_df, use_container_width=True)

# Section: Testimonials (Page 2)
elif section == "Testimonials":
    st.header("Customer Testimonials")
    testi_df = df[['text']].head(10).rename(columns={'text': 'Customer reviews'})
    testi_df.index = range(1, len(testi_df) + 1)
    st.table(testi_df)

# Section: Reviews (Page 3)
elif section == "Reviews - with sentiment":
    st.header("2023 Review Analysis")

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    selected_month_name = st.select_slider("Filter by Month (2023):", options=months)
    
    month_int = months.index(selected_month_name) + 1
    filtered_df = df[df['date'].dt.month == month_int].copy()

    if not filtered_df.empty:
        with st.spinner(f"Analyzing sentiment for {selected_month_name}..."):
            texts = filtered_df['text'].tolist()
            results = sentiment_analyzer(texts)
            
            filtered_df['Sentiment'] = [res['label'] for res in results]
            filtered_df['Confidence'] = [res['score'] for res in results]

        st.subheader(f"Sentiment Distribution: {selected_month_name} 2023")
        
        chart_data = filtered_df.groupby('Sentiment').agg(
            Count=('Sentiment', 'count'),
            Avg_Score=('Confidence', 'mean')
        ).reset_index()
        
        chart_data['Average Confidence Score'] = chart_data['Avg_Score'].round(3)

        fig = px.bar(
            chart_data, 
            x='Sentiment', 
            y='Count',
            color='Sentiment',
            color_discrete_map={'POSITIVE': 'green', 'NEGATIVE': 'red'},
            hover_data={
                'Average Confidence Score': True, 
                'Sentiment': True, 
                'Count': True,
                'Avg_Score': False
            },
            title=f"Total Reviews: {len(filtered_df)}"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.metric("Model Prediction Confidence", f"{filtered_df['Confidence'].mean():.2%}")

        st.subheader("Key Conversation Topics (Word Cloud)")
        text_combined = " ".join(filtered_df['text'])
        wordcloud = WordCloud(background_color="white", width=800, height=400).generate(text_combined)
        
        fig_wc, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig_wc)

        st.subheader("Raw Filtered Data")
        
        # Apply cleaning logic to 'title'
        filtered_df['Product Name'] = filtered_df['title'].apply(clean_product_name)
        
        # Format the date to remove the hour part
        filtered_df['Date Only'] = filtered_df['date'].dt.date
        
        # Select and rename columns for display
        display_df = filtered_df[['Date Only', 'Product Name', 'text', 'Sentiment', 'Confidence']].rename(
            columns={'Date Only': 'Date', 'text': 'Review'}
        )
        display_df.index = range(1, len(display_df) + 1)
        st.dataframe(display_df, use_container_width=True)

    else:
        st.warning(f"No reviews found for {selected_month_name} 2023.")