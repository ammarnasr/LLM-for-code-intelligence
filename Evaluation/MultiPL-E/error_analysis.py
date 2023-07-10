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

def load_json_file(file_path):
    """Loads a JSON file.
    Args:
        file_path (str): Path to JSON file.
    Returns:
        dict: JSON file contents.
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def get_df_from_data(data):
    propmpt = data['prompt']
    language = data['language']
    temperature = data['temperature']
    top_p = data['top_p']
    max_new_tokens = data['max_new_tokens']
    stop_tokens = data['stop_tokens']
    results = data['results']
    program = []
    timestamp = []
    stdout = []
    stderr = []
    exit_code = []
    status = []
    for result in results:
        program.append(result['program'])
        timestamp.append(result['timestamp'])
        stdout.append(result['stdout'])
        stderr.append(result['stderr'])
        exit_code.append(result['exit_code'])
        status.append(result['status'])
    prompt = [propmpt] * len(program)
    language = [language] * len(program)
    temperature = [temperature] * len(program)
    top_p = [top_p] * len(program)
    max_new_tokens = [max_new_tokens] * len(program)
    stop_tokens = [stop_tokens] * len(program)


    df = pd.DataFrame({
        'prompt': propmpt,
        'language': language,
        'temperature': temperature,
        'top_p': top_p,
        'max_new_tokens': max_new_tokens,
        'stop_tokens': stop_tokens,
        'program': program,
        'timestamp': timestamp,
        'stdout': stdout,
        'stderr': stderr,
        'exit_code': exit_code,
        'status': status
    })
    return df

def concat_two_df(df1, df2):
    return pd.concat([df1, df2])

def get_df_from_files(files):
    df = pd.DataFrame()
    for file in files:
        data = load_json_file(file)
        df = concat_two_df(df, get_df_from_data(data))
    return df

def select_columns(df, columns):
    return df[columns]

def get_value_counts(df, column):
    return df[column].value_counts()

def get_folders_in_dir(dir_path):
    """Returns a list of folders in a directory.
    Args:
        dir_path (str): Path to directory.
    Returns:
        list: List of folder paths.
    """
    folders = []
    for folder in os.listdir(dir_path):
        if os.path.isdir(os.path.join(dir_path, folder)):
            folders.append(os.path.join(dir_path, folder))
    return folders

def find_strings_in_df(df, column, strings):
    """Finds rows in a dataframe that contain a string in a column.
    Args:
        df (pandas.DataFrame): Dataframe.
        column (str): Column to search.
        strings (list): List of strings to search for.
    Returns:
        pandas.DataFrame: Dataframe with rows that contain a string in a column.
    """
    return df[df[column].str.contains('|'.join(strings))]

if __name__ == "__main__":
    init_page()
    parent_dir = './temp'
    all_strings = [
        "error: ';' expected",
        " java.lang.AssertionError",
        " ArrayList<"
        ]

    folders = get_folders_in_dir(parent_dir)
    java_folders = [folder for folder in folders if 'java' in folder]
    


    dirs = st.sidebar.multiselect('Select a folder', java_folders)
    strings = st.sidebar.multiselect('Select a string', all_strings)

    counts_dict = {
        'folder': [],
        'string': [],
        'count': []
    }

    for dir in dirs:
        ext = '.results.json'
        files = get_files_in_dir(dir, ext)
        df = get_df_from_files(files)
        for string in strings:
            s = [string]
            string_df = find_strings_in_df(df, 'stderr', s)
            counts_dict['folder'].append(dir)
            counts_dict['string'].append(string)
            counts_dict['count'].append(len(string_df))
    
    counts_df = pd.DataFrame(counts_dict)
    sns.barplot(x='folder', y='count', hue='string', data=counts_df)
    plt.xticks(rotation=45)
    st.pyplot()

    
    target_dir = st.selectbox('Select a folder', dirs)
    ext = '.results.json'
    files = get_files_in_dir(target_dir, ext)
    df = get_df_from_files(files)
    target_strings = st.multiselect('Select a string', strings, key='target_strings')
    target_df = find_strings_in_df(df, 'stderr', target_strings)
    target_df = select_columns(target_df, ['program', 'stderr'])
    target_index = st.number_input('Select an index', min_value=0, max_value=len(target_df)-1, value=0, step=1)
    target_df = target_df.iloc[target_index]
    target_program = target_df['program']
    st.code(target_program, language='java')
    st.dataframe(target_df)