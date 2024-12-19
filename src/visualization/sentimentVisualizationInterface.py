
class SentimentVisualizationInterface():
    def plot_sentiment(self):
        """
        Generate a bar plot to compare sentiment scores between virtual and human influencers.

        The plot shows average sentiment scores for different sentiment types (e.g., Positive,
        Negative, Neutral) with a comparison between virtual and human influencer sources.
        """
        raise NotImplementedError

    def generate_wordclouds(self):
        """
        Generate and display word clouds for positive and negative text data.

        Separate word clouds are created for:
        - Positive words from virtual influencers
        - Positive words from human influencers
        - Negative words from virtual influencers
        - Negative words from human influencers
        """
        raise NotImplementedError


