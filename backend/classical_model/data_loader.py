
import os
import pandas as pd

def load_json_to_dataframe(directory, filename):
    """
    """
   
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file '{filename}' does not exist in the directory '{directory}'.")
    try:
        dataframe = pd.read_json(filepath, lines=True)
        return dataframe
    except ValueError as e:
        raise ValueError(f"Failed to load JSON file '{filename}': {e}")