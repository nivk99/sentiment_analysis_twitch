import os
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import download
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import numpy as np
from src.sentiment.sentimentAnalysisInterface import SentimentAnalysisInterface
from src.visualization.sentiment_visualization import SentimentVisualization


# Download required NLTK resources if not already downloaded
# download('vader_lexicon')


class SentimentAnalysis(SentimentAnalysisInterface):
    def __init__(self, path: str = "output_file_twitch.csv"):
        """
        Initialize the SentimentAnalysis class.

        :param path: Path to the input CSV file containing text data.
        """
        self._sia = SentimentIntensityAnalyzer()  # Initialize sentiment analysis tool
        self._file_path = os.path.join(os.getcwd(), "data", path)  # Construct the full file path
        self._df = self.load_data()  # Load data from CSV
        self._virtual_scores = None
        self._real_scores = None
        self.sentiment_comparison = None

        # Perform initial analysis and hypothesis testing
        self.analysis()
        self.hypothesis_testing()
        SentimentVisualization(self)

    def load_data(self):
        """
        Load the dataset from the specified CSV file path.
        """
        try:
            df = pd.read_csv(self._file_path)
            df["Cleaned_Text"] = df["Cleaned_Text"].fillna("").astype(str)  # Ensure text data is formatted properly
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {self._file_path} does not exist.")
        except pd.errors.EmptyDataError:
            raise ValueError(f"The file {self._file_path} is empty.")
        except Exception as e:
            raise Exception(f"An error occurred while loading the data: {e}")

    def analysis(self):
        """
        Perform sentiment analysis on text data and calculate sentiment scores.
        """
        # Calculate sentiment scores for each text entry
        self._df['Positive'] = self._df["Cleaned_Text"].apply(lambda x: self._sia.polarity_scores(x)['pos'])
        self._df['Neutral'] = self._df["Cleaned_Text"].apply(lambda x: self._sia.polarity_scores(x)['neu'])
        self._df['Negative'] = self._df["Cleaned_Text"].apply(lambda x: self._sia.polarity_scores(x)['neg'])
        self._df['Compound'] = self._df["Cleaned_Text"].apply(lambda x: self._sia.polarity_scores(x)['compound'])

        # Display sentiment scores for each source
        print(self._df[['Source', 'Positive', 'Neutral', 'Negative', 'Compound']])

        # Perform comparative analysis of sentiment by source (virtual vs. human influencers)
        self.sentiment_comparison = self._df.groupby('Source')[['Positive', 'Neutral', 'Negative', 'Compound']].mean()
        print("\nAverage Sentiment Comparison: vi and HI")
        print( self.sentiment_comparison, "\n")

        # Separate sentiment scores based on the source
        self._virtual_scores = self._df[self._df['Source'] == 'VI']
        self._real_scores = self._df[self._df['Source'] == 'HI']

    def hypothesis_testing(self):
        """
        Perform statistical tests to validate hypotheses based on sentiment scores.
        """

        def cohen_d(x1, x2):
            # Calculate the weighted standard deviation
            pooled_std = np.sqrt(((len(x1) - 1) * np.var(x1, ddof=1) + (len(x2) - 1) * np.var(x2, ddof=1)) /
                                 (len(x1) + len(x2) - 2))
            # Calculating effect size
            return (np.mean(x1) - np.mean(x2)) / pooled_std

        print("Sample Sizes:")
        print(f"Virtual Influencers (VI): {len(self._virtual_scores)}")
        print(f"Human Influencers (HI): {len(self._real_scores)}\n")

        # Hypothesis 1: Are emotional responses to virtual influencers more negative?
        d_negative = cohen_d(self._virtual_scores['Negative'], self._real_scores['Negative'])
        t_stat_neg, p_value_neg = stats.ttest_ind(self._virtual_scores['Negative'], self._real_scores['Negative'])
        print(
            f"Hypothesis 1 - T-statistic (Negative Sentiment): {t_stat_neg}, P-value: {p_value_neg}, VI Variance: {self._virtual_scores['Negative'].var()},HI Variance: {self._real_scores['Negative'].var()},Cohen's d for Negative Sentiment: {d_negative}")
        if p_value_neg < 0.05:
            print(
                "Hypothesis 1: Emotional responses to posts by virtual influencers are significantly more negative.\n")
        else:
            print(
                "Hypothesis 1: No significant difference in negative sentiment between virtual and human influencers.\n")

        # Hypothesis 2: Do users exhibit higher positive sentiment for human influencers?
        d_Positive = cohen_d(self._virtual_scores['Positive'], self._real_scores['Positive'])
        t_stat_pos, p_value_pos = stats.ttest_ind(self._virtual_scores['Positive'], self._real_scores['Positive'])
        print(
            f"Hypothesis 2 - T-statistic (Positive Sentiment): {t_stat_pos}, P-value: {p_value_pos}, VI Variance: {self._virtual_scores['Positive'].var()},HI Variance: {self._real_scores['Positive'].var()},Cohen's d for Positive Sentiment: {d_Positive}")
        if p_value_pos < 0.05:
            print("Hypothesis 2: Users exhibit significantly higher positive sentiment for human influencers.\n")
        else:
            print(
                "Hypothesis 2: No significant difference in positive sentiment between human and virtual influencers.\n")

        # Hypothesis 3: Is there a significant difference in overall sentiment (Compound)?
        d_Compound = cohen_d(self._virtual_scores['Compound'], self._real_scores['Compound'])
        t_stat_compound, p_value_compound = stats.ttest_ind(self._virtual_scores['Compound'],
                                                            self._real_scores['Compound'])
        print(
            f"Hypothesis 3 - T-statistic (Compound Sentiment): {t_stat_compound}, P-value: {p_value_compound}, VI Variance: {self._virtual_scores['Compound'].var()},HI Variance: {self._real_scores['Compound'].var()},Cohen's d for Compound Sentiment: {d_Compound}")
        if p_value_compound < 0.05:
            print("Hypothesis 3: Significant difference in overall emotional responses between influencers.\n")
        else:
            print("Hypothesis 3: No significant difference in overall emotional responses between influencers.\n")
