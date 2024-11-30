from src.clean.clean_data_frame_twitch import CleanDataFrameTwitch
from src.data_frame.data_frame_twitch import DataFrameTwitch
from src.sentiment.sentiment_analysis import SentimentAnalysis
from src.visualization.sentiment_visualization import SentimentVisualization

PATH: str = "output_file_twitch.csv"
if __name__ == '__main__':
   # DataFrameTwitch(PATH)
    CleanDataFrameTwitch(PATH)
    SentimentAnalysis(PATH)
   # SentimentVisualization()
