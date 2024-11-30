from src.sentiment.sentimentAnalysisInterface import SentimentAnalysisInterface
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import download
from scipy import stats
import os

import matplotlib.pyplot as plt
import seaborn as sns


# Download NLTK resources
# download('vader_lexicon')

class SentimentAnalysis(SentimentAnalysisInterface):

    def __init__(self, path: str = "output_file_twitch.csv"):
        # Create sentiment analysis tool
        self._sia = SentimentIntensityAnalyzer()
        self._virtual_scores = None
        self._real_scores = None
        self._file_path = os.getcwd() + "/data" + "/" + path
        self._df = pd.read_csv(self._file_path)
        self.analysis()
        self.hypothesis_testing()
        self.visual_analysis()

    def analysis(self):
        # Ensure 'Cleaned_Text' column is properly formatted as strings and handle missing values
        self._df['Cleaned_Text'] = self._df['Cleaned_Text'].fillna("").astype(str)

        # Calculate sentiment scores
        self._df['Positive'] = self._df['Cleaned_Text'].apply(lambda x: self._sia.polarity_scores(x)['pos'])
        self._df['Neutral'] = self._df['Cleaned_Text'].apply(lambda x: self._sia.polarity_scores(x)['neu'])
        self._df['Negative'] = self._df['Cleaned_Text'].apply(lambda x: self._sia.polarity_scores(x)['neg'])
        self._df['Compound'] = self._df['Cleaned_Text'].apply(lambda x: self._sia.polarity_scores(x)['compound'])

        # Print sentiment scores
        print(self._df[['Source', 'Positive', 'Neutral', 'Negative', 'Compound']].map(lambda x: x))

        # ניתוח השוואתי של רגשות בין דמויות וירטואליות ואמיתיות
        sentiment_comparison = self._df.groupby('Source')[['Positive', 'Neutral', 'Negative', 'Compound']].mean()

        # sentiment_comparison = df.groupby('Source')['Compound'].mean()

        print("\nAverage Sentiment Comparison:")
        print(sentiment_comparison, "\n")

        # Separate scores by source
        self._virtual_scores = self._df[self._df['Source'] == 'VI']
        self._real_scores = self._df[self._df['Source'] == 'HI']

    def hypothesis_testing(self):
        # Hypothesis 1: Are emotional responses to virtual influencers more negative?
        t_stat_neg, p_value_neg = stats.ttest_ind(self._virtual_scores['Negative'], self._real_scores['Negative'])
        print(f"Hypothesis 1 - T-statistic (Negative Sentiment): {t_stat_neg}, P-value: {p_value_neg}")
        if p_value_neg < 0.05:
            print(
                "Hypothesis 1: Emotional responses to posts by virtual influencers are significantly more negative than "
                "those by human influencers.\n")
        else:
            print(
                "Hypothesis 1: No significant difference in negative sentiment between virtual and human influencers.\n")

        # Hypothesis 2: Do users exhibit higher positive sentiment for human influencers?
        t_stat_pos, p_value_pos = stats.ttest_ind(self._virtual_scores['Positive'], self._real_scores['Positive'])
        print(f"Hypothesis 2 - T-statistic (Positive Sentiment): {t_stat_pos}, P-value: {p_value_pos}")
        if p_value_pos < 0.05:
            print(
                "Hypothesis 2: Users exhibit significantly higher positive sentiment for content by human influencers "
                "compared to virtual influencers.\n")
        else:
            print(
                "Hypothesis 2: No significant difference in positive sentiment between human and virtual influencers.\n")

        # Hypothesis 3: Is there a significant difference in overall sentiment (Compound) between virtual and human influencers?
        t_stat_compound, p_value_compound = stats.ttest_ind(self._virtual_scores['Compound'],
                                                            self._real_scores['Compound'])
        print(f"Hypothesis 3 - T-statistic (Compound Sentiment): {t_stat_compound}, P-value: {p_value_compound}")
        if p_value_compound < 0.05:
            print("Hypothesis 3: There is a significant difference in overall emotional responses (Compound sentiment) "
                  "between virtual and human influencers.\n")
        else:
            print("Hypothesis 3: No significant difference in overall emotional responses between virtual and human "
                  "influencers.\n")

    def visual_analysis(self):
        """
        Perform visual analysis of sentiment scores.
        """
        # Set up the visualization styles
        sns.set(style="whitegrid")

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot for positive sentiment
        sns.boxplot(x="Source", y="Positive", data=self._df, ax=axes[0, 0])
        axes[0, 0].set_title("Positive Sentiment by Source")

        # Plot for negative sentiment
        sns.boxplot(x="Source", y="Negative", data=self._df, ax=axes[0, 1])
        axes[0, 1].set_title("Negative Sentiment by Source")

        # Plot for neutral sentiment
        sns.boxplot(x="Source", y="Neutral", data=self._df, ax=axes[1, 0])
        axes[1, 0].set_title("Neutral Sentiment by Source")

        # Plot for compound sentiment
        sns.boxplot(x="Source", y="Compound", data=self._df, ax=axes[1, 1])
        axes[1, 1].set_title("Compound Sentiment by Source")

        # Adjust layout
        plt.tight_layout()

        # Show plots
        plt.show()


