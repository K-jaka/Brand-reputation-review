# 2023 Brand Reputation Monitor

An AI-powered dashboard that analyzes customer sentiment and brand reputation from scraped product reviews. Built with Streamlit and Hugging Face Transformers (DistilBERT) as part of a Data Mining course.

## About

The project scrapes product reviews from a public API, filters them to 2023, and runs sentiment analysis on each review using a pre-trained DistilBERT model. The results are displayed in an interactive dashboard with monthly filtering, sentiment charts, and word clouds.

## Features

- **Sentiment Analysis** — classifies reviews as Positive or Negative using DistilBERT
- **Monthly Filtering** — explore sentiment trends across all 12 months of 2023
- **Word Clouds** — visualize the most common topics per month
- **Product Browser** — clean, deduplicated list of all reviewed products
- **Customer Testimonials** — displays the 10 most recent reviews

## Setup

Requires Python 3.8+.
```bash
# 1. Clone the repository
git clone https://github.com/K-jaka/Brand-reputation-review

# 2. Install dependencies
pip3 install -r requirements.txt

# 3. Run the scraper to generate the data file
python scraper.py

# 4. Launch the dashboard
streamlit run app.py
```

The app will open at `http://localhost:8501`. The first run will be slow as DistilBERT (~250MB) downloads and caches automatically.

## Project Structure
```
├── app.py                  # Streamlit dashboard
├── scraper.py              # Review scraper
├── requirements.txt        # Dependencies
└── README.md
```
