class DataFrameTwitchInterface:
    """
    Abstract base class (interface) for processing Twitch data into a DataFrame.
    Ensures that derived classes implement the required methods.
    """

    def process_text_file(self, input_file: str, source: str, player: str) -> None:
        """
        Process a single text file to extract data.

        :param input_file: Path to the text file.
        :param source: Source name extracted from the filename.
        :param player: Player name extracted from the filename.
        """
        raise NotImplementedError

    def create_data_frame(self) -> None:
        """
        Create a DataFrame from the processed data and save it to a CSV file.
        """
        raise NotImplementedError

    def process_all_text_files_in_directory(self) -> None:
        """
        Process all text files in a directory and store the data.
        """
        raise NotImplementedError
