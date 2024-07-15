import os
from collections import Counter

from django.conf import settings
from matplotlib import pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from wordcloud import WordCloud

from utils.database import select_news_by_id


# import nltk
# nltk.download('vader_lexicon')


database_path = "../../stocks421.db"


def generate_wordcloud(database_path, id, store_path, top_n=300, negative_weight=20):
    stop_words = ENGLISH_STOP_WORDS.union({
        "china", "said", "hong", "kong", "hk", "the", "share", "shares", 'cent', 'my', 'that', 'at', 'with', 'me', 'do',
        'have', 'this', 'be', 'I', 'not', 'or', 'are', 'your', 'if', 'can', 'but', 'was', 'had', 'per', 'other', 'has',
        'cent,', 'cent.', 'was', 'were', 'while', ',', 'some', 'when', 'market'
    })
    sid = SentimentIntensityAnalyzer()

    news = select_news_by_id(database_path, id)
    content = news['content'][0]

    # Tokenize the content (basic tokenizer, adjust as needed)
    tokens = content.split()

    # Calculate sentiment scores for each token
    word_sentiment_scores = {}
    word_frequency = Counter(tokens)

    for token in tokens:
        # Skip stop words
        if token.lower() in stop_words:
            continue

        # Calculate sentiment score
        scores = sid.polarity_scores(token)
        score = scores['compound']  # Using compound score as the overall sentiment score

        # Adjust sentiment score based on negative words
        original_score = score  # Keep the original score for color calculation
        if score < 0:  # If the word has negative sentiment
            score = -score
            score *= negative_weight

        # Weighted sentiment score based on frequency
        weighted_score = score * word_frequency[token]

        # Update sentiment score for the token
        if token in word_sentiment_scores:
            word_sentiment_scores[token] += weighted_score
        else:
            word_sentiment_scores[token] = weighted_score

    # Sort words by sentiment scores and get top n words
    sorted_word_sentiments = sorted(word_sentiment_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_words = {word: score for word, score in sorted_word_sentiments}

    # Define a color function
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        original_score = sid.polarity_scores(word)['compound']
        if original_score >= 0:
            red = min(255, 255 - int((original_score - 0.1) * 2 * 50))
            green = min(255, 105 - int(original_score * 50))
            blue = min(255, 180 - int(original_score * 50))
            color = f"rgb({red}, {green}, {blue})"  # brighter pink shades
        else:
            blue = min(255, 205 - int(abs(original_score * 5 - 0.3) / 5 * 50))
            green = min(255, 100 - int(abs(original_score * 2 - 0.1) / 2 * 50))
            red = min(255, 50 - int(abs(original_score - 0.1) * 50))
            color = f"rgb({red}, {green}, {blue})"  # brighter blue shades
        return color

    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          color_func=color_func).generate_from_frequencies(top_words)

    upload_dir = store_path
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    file_path = os.path.join(upload_dir, f'wordcloud_{id}.png')

    # Display word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(news['title'][0])
    # plt.show()
    plt.savefig(file_path, dpi=500)

    return file_path


# generate_wordcloud(database_path, 1)
