from src.data_frame.dataFrameTwitchInterface import DataFrameTwitchInterface
import os
import pandas as pd


class DataFrameTwitch(DataFrameTwitchInterface):
    def __init__(self, path: str = "output_file_twitch.csv"):
        # Initialize attributes for storing data from text files
        self._times: list[str] = []
        self._texts: list[str] = []
        self._names: list[str] = []
        self._sources: list[str] = []
        self._player: list[str] = []

        # Paths for input directory and output file
        self._output_file_twitch: str = os.getcwd() + "/data/" + path
        self._input_file_path: str = os.getcwd() + "/data"
        print(self._input_file_path)

        # Process all text files in the directory
        self.process_all_text_files_in_directory()

    def process_text_file(self, input_file: str, source: str, player: str) -> None:
        """
        Reads a text file, processes its content, and stores the data in class attributes.

        :param input_file: Path to the input text file.
        :param source: Source name extracted from the filename.
        :param player: Player name extracted from the filename.
        """
        with open(input_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # Process each line in the file
        current_time = None
        for line in lines:
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            # Check if the line contains a timestamp (assume format 'hh:mm')
            if ':' in line[:5]:
                current_time = line
            else:
                # Split the line into name and text content
                if ':' in line:
                    name, text = line.split(':', 1)
                else:
                    name, text = "Unknown", line  # Handle cases without a colon
                self._times.append(current_time)
                self._texts.append(text.strip())
                self._names.append(name.strip())
                self._sources.append(source)
                self._player.append(player)

    def create_data_frame(self) -> None:
        """
        Creates a DataFrame from the stored data and saves it as a CSV file.
        """
        # Create a DataFrame
        df = pd.DataFrame({
            'Player': self._player,
            'Source': self._sources,
            'Time': self._times,
            'Name': self._names,
            'Text': self._texts
        })

        # Save the DataFrame to a CSV file
        df.to_csv(self._output_file_twitch, index=False)

    def process_all_text_files_in_directory(self) -> None:
        """
        Processes all text files in the specified input directory.
        """
        for filename in os.listdir(self._input_file_path):
            if filename.endswith('.txt'):  # Check if the file is a text file
                input_file = os.path.join(self._input_file_path, filename)

                # Extract the base name (without extension) and split into source and player
                base_name = os.path.splitext(filename)[0]
                try:
                    source, player = base_name.split('_', 1)  # Split by underscore
                    print(f"Processing file: {filename}")
                    print(f"Extracted names: source = {source}, player = {player}")
                except ValueError:
                    print(f"Error: Could not split filename '{base_name}' into two parts.")
                    continue

                # Process the file
                self.process_text_file(input_file, source, player)

        # Create the DataFrame and save it as a CSV
        self.create_data_frame()


