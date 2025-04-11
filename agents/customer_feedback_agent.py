from textblob import TextBlob

class CustomerFeedbackAgent:
    def analyze_sentiment(self, review_text):
        analysis = TextBlob(review_text)
        polarity = analysis.sentiment.polarity
        if polarity > 0.1:
            return "Positive"
        elif polarity < -0.1:
            return "Negative"
        else:
            return "Neutral"
