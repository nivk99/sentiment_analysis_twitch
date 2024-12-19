
class cleanDataFrameTwitchInterface:

    class CleanDataFrameTwitchInterface:
        """
        Abstract Base Class for CleanDataFrameTwitch.
        Defines the required methods for cleaning and processing data.
        """

        def ensure_nltk_resources(self):

            """
            Ensure required NLTK resources are downloaded.
            """
            raise NotImplementedError


        def create_emoticon_dict(self):
            """
            Create a dictionary mapping common emoticons to their sentiment descriptions.
            """
            raise NotImplementedError

        def replace_emoticons(self, text):
            """
            Replace emoticons in the text with their sentiment description.
            """
            raise NotImplementedError

        def replace_emojis(self, text):
            """
            Replace emojis in the text with their sentiment description using emoji.demojize().
            """
            raise NotImplementedError

        def add_comma_space(self, text):
            """
            Add a space after commas if missing.
            """
            raise NotImplementedError

        def clean_text_for_sentiment(self, text):
            """
            Clean text for sentiment analysis.
            """
            raise NotImplementedError

        def process_csv_and_clean_text(self):
            """
            Process CSV file, clean its content, and save the result.
            """
            raise NotImplementedError
