from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import pickle
import re
from nltk.tokenize import RegexpTokenizer
from nltk import FreqDist
from nltk.stem import PorterStemmer
from datetime import datetime

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('data/assignment3.csv')

# Load the trained model and vectorizer
with open('models/lr_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load stopwords
with open('stopwords_en.txt', 'r') as f:
    stopwords = set(word.strip() for word in f.readlines())

# Initialize tokenizer and stemmer
pattern = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"
tokenizer = RegexpTokenizer(pattern)
stemmer = PorterStemmer()
# Calculate rare words (hapaxes) and top 20 most common words from training data
all_tokens = []
for review_text in df['Review Text'].fillna(''):
    text_clean = review_text.lower()
    text_clean = re.sub(r"\s+", " ", text_clean).strip()
    tokens = tokenizer.tokenize(text_clean)
    all_tokens.extend(tokens)

# Also include title tokens
for title in df['Title'].fillna(''):
    title_clean = title.lower()
    title_clean = re.sub(r"\s+", " ", title_clean).strip()
    tokens = tokenizer.tokenize(title_clean)
    all_tokens.extend(tokens)

# Calculate frequencies
term_freq = FreqDist(all_tokens)
rare_words = set(term_freq.hapaxes())  # Words that appear only once
top_20_words = set([word for word, freq in term_freq.most_common(20)])

# Preprocessing function (same as in notebook)
def preprocess_text(text):
    """Preprocess text using the same pipeline as training"""
    if pd.isna(text) or text == '':
        return ''

    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    tokens = tokenizer.tokenize(text)

    # Filter tokens: remove short words, stopwords, rare words, and top 20 common words
    filtered = [
        w for w in tokens
        if len(w) >= 2
        and w not in stopwords
        and w not in rare_words
        and w not in top_20_words
    ]

    return " ".join(filtered)

# Search function with stemming
def search_items(keyword):
    """Search for clothing items based on keyword with stemming support"""
    if not keyword:
        return df

    keyword = keyword.lower().strip()
    keyword_stem = stemmer.stem(keyword)

    # Helper function to check if stemmed words match
    def contains_stem(text, stem):
        if pd.isna(text):
            return False
        text_lower = str(text).lower()
        # Tokenize the text and check if any stemmed token matches the search stem
        tokens = tokenizer.tokenize(text_lower)
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
        return stem in stemmed_tokens

    # Search in Class Name, Department Name, Division Name, Title, and Clothes Title
    # Only use stemmed match for consistent results
    mask = (
        df['Class Name'].apply(lambda x: contains_stem(x, keyword_stem)) |
        df['Department Name'].apply(lambda x: contains_stem(x, keyword_stem)) |
        df['Division Name'].apply(lambda x: contains_stem(x, keyword_stem)) |
        df['Title'].apply(lambda x: contains_stem(x, keyword_stem)) |
        df['Clothes Title'].apply(lambda x: contains_stem(x, keyword_stem))
    )

    return df[mask]

@app.route('/')
def index():
    """Home page with search functionality and all items display"""
    query = request.args.get('q', '')

    # Get all unique items
    all_items = df.drop_duplicates(subset=['Clothing ID'])

    # Apply search filter if query exists
    if query:
        results = search_items(query)
        filtered_items = results.drop_duplicates(subset=['Clothing ID'])
    else:
        filtered_items = all_items

    count = len(filtered_items)

    return render_template('index.html',
                         query=query,
                         results=filtered_items.to_dict('records'),
                         count=count)

@app.route('/search')
def search():
    """Search results page - redirect to index with query parameter"""
    query = request.args.get('q', '')

    if query:
        return redirect(url_for('index', q=query))
    else:
        return redirect(url_for('index'))

@app.route('/item/<int:clothing_id>')
def item_detail(clothing_id):
    """Item detail page showing all reviews for this clothing item"""
    # Get all reviews for this item
    item_reviews = df[df['Clothing ID'] == clothing_id]

    if len(item_reviews) == 0:
        return "Item not found", 404

    # Get item information from first review
    item_info = item_reviews.iloc[0]

    # Sort reviews: newest first (reviews with 'Created At' field first, then by index descending)
    # This ensures newly added reviews appear at the top
    item_reviews = item_reviews.sort_index(ascending=False)

    return render_template('item_detail.html',
                         item=item_info.to_dict(),
                         reviews=item_reviews.to_dict('records'),
                         clothing_id=clothing_id)

@app.route('/item/<int:clothing_id>/add_review', methods=['GET', 'POST'])
def add_review(clothing_id):
    """Add a new review for an item"""
    global df

    # Get item information
    item_reviews = df[df['Clothing ID'] == clothing_id]

    if len(item_reviews) == 0:
        return "Item not found", 404

    item_info = item_reviews.iloc[0]

    if request.method == 'POST':
        # Get form data
        title = request.form.get('title', '')
        review_text = request.form.get('review_text', '')
        rating = int(request.form.get('rating', 3))
        age = int(request.form.get('age', 30))

        # Preprocess title and review
        processed_title = preprocess_text(title)
        processed_review = preprocess_text(review_text)
        combined_text = processed_review + ' ' + processed_title

        # Make prediction
        if combined_text.strip():
            features = vectorizer.transform([combined_text])
            prediction = model.predict(features)[0]
            prediction_proba = model.predict_proba(features)[0]
        else:
            prediction = 0
            prediction_proba = [0.5, 0.5]

        # Check if user wants to override
        if 'override' in request.form:
            final_recommendation = int(request.form.get('recommendation', prediction))

            # Create new review dictionary
            new_review = {
                'Clothing ID': clothing_id,
                'Age': age,
                'Title': title,
                'Review Text': review_text,
                'Rating': rating,
                'Recommended IND': final_recommendation,
                'Positive Feedback Count': 0,
                'Division Name': item_info['Division Name'],
                'Department Name': item_info['Department Name'],
                'Class Name': item_info['Class Name'],
                'Clothes Title': item_info['Clothes Title'],
                'Clothes Description': item_info['Clothes Description'],
                'Processed Tokens': combined_text,
                'Created At': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            # Append to dataframe
            df = pd.concat([df, pd.DataFrame([new_review])], ignore_index=True)

            # Save to CSV file
            df.to_csv('data/assignment3.csv', index=False)

            # Redirect to show the newly created review
            return redirect(url_for('item_detail', clothing_id=clothing_id))

        # Show prediction to user for confirmation
        return render_template('add_review.html',
                             item=item_info.to_dict(),
                             clothing_id=clothing_id,
                             form_data={
                                 'title': title,
                                 'review_text': review_text,
                                 'rating': rating,
                                 'age': age
                             },
                             prediction=prediction,
                             prediction_proba=prediction_proba,
                             show_prediction=True)

    # GET request - show empty form
    return render_template('add_review.html',
                         item=item_info.to_dict(),
                         clothing_id=clothing_id,
                         show_prediction=False)

if __name__ == '__main__':
    app.run(port=5000)
