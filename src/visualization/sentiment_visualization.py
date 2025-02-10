from src.sentiment.sentimentAnalysisInterface import SentimentAnalysisInterface
from src.visualization.sentimentVisualizationInterface import SentimentVisualizationInterface
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
from collections import Counter


class SentimentVisualization(SentimentVisualizationInterface):

    """
    A class for visualizing sentiment analysis results of text data associated with
    virtual and human influencers. This includes generating comparative sentiment
    bar plots and word clouds for positive and negative sentiment categories.

    Attributes:
        _sentimect (SentimentAnalysisInterface): An instance of a sentiment analysis interface
            that provides the necessary sentiment data for visualization.
    """

    def __init__(self, sentimect: SentimentAnalysisInterface):
        self._sentimect = sentimect
        self.plot_sentiment()
        self.generate_wordclouds()
        self.plot_word()

    def plot_sentiment(self):
        # Plot sentiment comparison
        sentiment_comparison = self._sentimect.sentiment_comparison.reset_index()
        plt.figure(figsize=(10, 6))
        sns.barplot(data=pd.melt(sentiment_comparison, id_vars='Source'), x='variable', y='value', hue='Source')
        plt.title('Sentiment Comparison between Virtual and Human Influencers')
        plt.xlabel('Sentiment Type')
        plt.ylabel('Average Score')
        plt.legend(title='Source')
        plt.show()

    def generate_wordclouds(self):
        """
        Generate comparison and commonality word clouds for text data from virtual and human influencers.
        """

        # Positive words for VI and HI
        positive_vi_text = " ".join(
            self._sentimect._virtual_scores[self._sentimect._virtual_scores['Positive'] > 0.5]["Cleaned_Text"].tolist())
        positive_hi_text = " ".join(
            self._sentimect._real_scores[self._sentimect._real_scores['Positive'] > 0.5]["Cleaned_Text"].tolist())

        # Negative words for VI and HI
        negative_vi_text = " ".join(
            self._sentimect._virtual_scores[self._sentimect._virtual_scores['Negative'] > 0.5]["Cleaned_Text"].tolist())
        negative_hi_text = " ".join(
            self._sentimect._real_scores[self._sentimect._real_scores['Negative'] > 0.5]["Cleaned_Text"].tolist())

        # Positive Word Clouds
        positive_cloud_vi = WordCloud(stopwords=STOPWORDS, background_color="white", colormap="Greens").generate(
            positive_vi_text)
        positive_cloud_hi = WordCloud(stopwords=STOPWORDS, background_color="white", colormap="Blues").generate(
            positive_hi_text)

        # Negative Word Clouds
        negative_cloud_vi = WordCloud(stopwords=STOPWORDS, background_color="white", colormap="Reds").generate(
            negative_vi_text)
        negative_cloud_hi = WordCloud(stopwords=STOPWORDS, background_color="white", colormap="Purples").generate(
            negative_hi_text)

        # Display Positive Word Clouds
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 2, 1)
        plt.imshow(positive_cloud_vi, interpolation='bilinear')
        plt.axis("off")
        plt.title("Positive Words - Virtual Influencers")

        plt.subplot(2, 2, 2)
        plt.imshow(positive_cloud_hi, interpolation='bilinear')
        plt.axis("off")
        plt.title("Positive Words - Human Influencers")

        # Display Negative Word Clouds
        plt.subplot(2, 2, 3)
        plt.imshow(negative_cloud_vi, interpolation='bilinear')
        plt.axis("off")
        plt.title("Negative Words - Virtual Influencers")

        plt.subplot(2, 2, 4)
        plt.imshow(negative_cloud_hi, interpolation='bilinear')
        plt.axis("off")
        plt.title("Negative Words - Human Influencers")

        plt.tight_layout()
        plt.show()


    def plot_word(self):
        """
        Generate bar plots to show the frequency of words for both Virtual and Human Influencers.
        """

        def get_word_frequencies(text):
            words = text.split()
            return Counter(words)

        # Text data
        vi_text = " ".join(self._sentimect._virtual_scores["Cleaned_Text"].tolist())
        hi_text = " ".join(self._sentimect._real_scores["Cleaned_Text"].tolist())

        # Word frequencies
        vi_frequencies = get_word_frequencies(vi_text).most_common(10)
        hi_frequencies = get_word_frequencies(hi_text).most_common(10)

        # Convert to DataFrame
        vi_df = pd.DataFrame(vi_frequencies, columns=['Word', 'Frequency'])
        hi_df = pd.DataFrame(hi_frequencies, columns=['Word', 'Frequency'])

        # Plot
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        sns.barplot(data=vi_df, x='Frequency', y='Word', palette="viridis")
        plt.title("Top 10 Words - Virtual Influencers")

        plt.subplot(1, 2, 2)
        sns.barplot(data=hi_df, x='Frequency', y='Word', palette="magma")
        plt.title("Top 10 Words - Human Influencers")

        plt.tight_layout()
        plt.show()