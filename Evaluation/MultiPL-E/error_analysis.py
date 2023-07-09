import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
sns.set(rc={'figure.figsize':(11.7,8.27)})


def init_page():
    st.title('Error Analysis')

def get_files_in_dir(dir_path, ext=None):
    """Returns a list of files in a directory, optionally filtered by extension.
    Args:
        dir_path (str): Path to directory.
        ext (str, optional): File extension to filter by. Defaults to None.
    Returns:
        list: List of file paths.
    """
    files = []
    for file in os.listdir(dir_path):
        if ext is None or file.endswith(ext):
            files.append(os.path.join(dir_path, file))
    return files



if __name__ == "__main__":
    init_page()
    dir = './temp/Baseline_java'
    ext = '.results.json'
    files = get_files_in_dir(dir, ext)
    
    st.write(files)