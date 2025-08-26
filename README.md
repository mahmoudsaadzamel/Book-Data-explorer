# Book Data Explorer

This is a Python application that enables you to ask natural language questions about book data (or any CSV file you upload). The application leverages Language Models (LLMs) to generate friendly, conversational responses based on the data in the CSV.

## Features

- Ask questions about book data in plain English
- Analyze book categories, ratings, prices, and availability
- Use default book dataset or upload your own CSV
- Get friendly, conversational responses
- View how the system found your answer

## How it works

The application reads the CSV file and processes the data using OpenAI LLMs together with Langchain Agents. The app includes special handling for common question types like description length comparisons and stock availability questions to ensure accurate responses.

The interface is built with Streamlit for a clean, user-friendly experience.

## Setup Instructions

### Local Development

1. Clone this repository to your local machine
2. Install the necessary dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
4. Run the application:
   ```
   streamlit run main.py
   ```

# 🔎 Scraper (optional, local-only)

This repo includes scraper.py to fetch a demo dataset from Books to Scrape (https://books.toscrape.com/
).
Scraping runs locally (not in the Space) for reliability and good citizenship.

Run the scraper
pip install -r requirements.txt
python3 scraper.py
# → writes books.csv in the repo root (default)


If your version of scraper.py supports flags, you can export to a custom location:
python3 scraper.py --out data/books.csv

What it writes 

When available, the CSV includes columns like:
title, category, price, rating, description
availability_text, availability_n, in_stock
product_url, image_url, upc
desc_word_count

Use the scraped file in the app

- Replace the committed books.csv with your new one or

- Keep the default and upload a different CSV from the UI

**Please be polite:** throttle requests and respect robots.txt. This is an educational demo.  

### Hugging Face Spaces Deployment

1. Create a new Space on Hugging Face
2. Choose Streamlit as the Space SDK
3. Upload the files from this repository
4. Add your OpenAI API key in the Space secrets:
   - Go to Settings > Repository Secrets
   - Add a secret named `OPENAI_API_KEY` with your API key as the value
5. The app will automatically deploy

## Usage

1. Choose whether to use the default book dataset or upload your own CSV
2. Type a question in the input field
3. Click "Ask Question" or press Enter
4. View the answer and optionally explore how the answer was found

Example questions:
- "Which category has the most books?"
- "What's the average price of Mystery books?"
- "Compare the average description length across the four categories."
- "Which category has the highest percentage of books marked as 'Out of stock'?"
```
