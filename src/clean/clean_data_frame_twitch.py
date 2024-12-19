import pandas as pd
import re
import string
import os
import nltk
from nltk.corpus import stopwords, opinion_lexicon, wordnet
from nltk.stem import PorterStemmer
import emoji


class CleanDataFrameTwitch:
    def __init__(self, path: str = "output_file_twitch.csv"):
        self._file_path = os.path.join(os.getcwd(), "data", path)
        self.stemmer = PorterStemmer()  # Initialize stemmer
        self.emoticon_dict = self.create_emoticon_dict()  # Initialize emoticon dictionary
        self.ensure_nltk_resources()


    def ensure_nltk_resources(self):
        """
        Ensure required NLTK resources are downloaded.
        If not, download them.
        """
        resources = ['stopwords', 'opinion_lexicon']
        for resource in resources:
            try:
                nltk.data.find(f'corpora/{resource}')
            except LookupError:
                print(f"{resource} not found. Downloading...")
                nltk.download(resource)

        self.process_csv_and_clean_text()

    def create_emoticon_dict(self):
        """
        Create a dictionary mapping common emoticons to their sentiment descriptions.
        """
        return {
            ":-)": "happy", ":)": "happy", ":-D": "very happy", "D:": "very happy",":D": "very happy",
            ":-(": "sad", ":(": "sad", ":-P": "playful", ":P": "playful",
            ";-)": "wink", ";)": "wink", ":-|": "neutral", ":|": "neutral"
        }
    def replace_emoticons(self, text):
        """
        Replace emoticons in the text with their sentiment description.
        """
        for emoticon, sentiment in self.emoticon_dict.items():
            text = text.replace(emoticon, sentiment)
        return text

    def replace_emojis(self, text):
        """
        Replace emojis in the text with their sentiment description using emoji.demojize().
        """
        return emoji.demojize(text, delimiters=(" ", " "))

    def add_comma_space(self, text):
        """
        Add a space after commas if missing.
        """
        return re.sub(r',(?=\S)', ', ', text)

    def clean_text_for_sentiment(self, text):
        """
        Clean text for sentiment analysis:
        - Remove URLs
        - Remove special characters and punctuation
        - Remove numbers
        - Convert to lowercase
        - Keep sentiment-related words (positive/negative)
        """
        if not isinstance(text, str):
            return ""  # Handle cases where text is not a string

        # Replace emojis and emoticons
        text = self.replace_emojis(text)
        text = self.replace_emoticons(text)

        # Add space after commas if missing
        text = self.add_comma_space(text)

        # Convert to lowercase
        text = text.lower()

        # Negative words to keep
        negative_words_keep = {"no", "not", "never", "barely", "hardly", "scarcely", "rarely", "without", "against",
                               "cannot"}

        # Load stopwords and sentiment lexicon
        stop_words = set(stopwords.words('english')) - negative_words_keep
        positive_words = set(opinion_lexicon.positive())
        negative_words = set(opinion_lexicon.negative())

        # Split text into words
        words = text.split()

        #  Filter words: keep if it's not a stopword or if it's in the sentiment lexicon self.stemmer.stem(word)
        filtered_words = [
            self.stemmer.stem(word) for word in words
           if word not in stop_words or word in positive_words or word in negative_words
        ]


        # Rejoin the words into a cleaned string
        text = ' '.join(filtered_words)


        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)

        # Remove special characters and punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove numbers
        text = re.sub(r'\d+', '', text)


        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()

        words = text.split()
        filtered_words = [
            self.stemmer.stem(word) for word in words
        ]

        text = ' '.join(filtered_words)


        return text

    def process_csv_and_clean_text(self):
        """
        Reads a CSV file, cleans the 'Text' column, and saves the result to a new file.
        """
        # Load the CSV file
        df = pd.read_csv(self._file_path)

        # Check if 'Text' column exists
        if 'Text' in df.columns:
            # Clean the 'Text' column
            df['Cleaned_Text'] = df['Text'].apply(lambda x: self.clean_text_for_sentiment(x))

            # Save the cleaned data
            df.to_csv(self._file_path, index=False)
            print(f"Cleaned text saved to: {self._file_path}")
        else:
            print("Error: 'Text' column not found in the file.")
