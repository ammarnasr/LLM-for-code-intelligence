import utils
import json
import streamlit as st
import os
import code_generation
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

st.set_page_config(
        page_title="Code Generation with Language Specific LoRa Models",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
st.title("Code Generation with Language Specific LoRa Models")


def get_json_files(dir):
    files = os.listdir(dir)
    files = [file for file in files if file.endswith('.results.json')]
    return files


def get_all_data(data_files, parent_dir, prompts, all_data=None):
    model_name = parent_dir.split('/')[-1]
    if all_data is None:
        all_data = {
            'prompt_number': [],
            'prompt_id': [],
            'prompt': [],
            'language': [],
            'temperature': [],
            'top_p': [],
            'max_new_tokens': [],
            'tests': [],
            'stop_tokens': [],
            'program': [],
            'timestamp': [],
            'stdout': [],
            'stderr': [],
            'exit_code': [],
            'status': [],
            'model_name': [],
        }
    for file in data_files:
        with open(parent_dir + '/' + file) as f:
            data = json.load(f)
        prompt = data['prompt']
        prompt_id = prompts[prompts['prompt_text'] == prompt]['prompt_id'].values[0]
        prompt_number = int(prompt_id.split('_')[1])
        language = data['language']
        temperature = data['temperature']
        top_p = data['top_p']
        max_new_tokens = data['max_new_tokens']
        tests = data['tests']
        stop_tokens = data['stop_tokens']
        results = data['results']
        for result in results:
            all_data['prompt_number'].append(prompt_number)
            all_data['prompt_id'].append(prompt_id)
            all_data['prompt'].append(prompt)
            all_data['language'].append(language)
            all_data['temperature'].append(temperature)
            all_data['top_p'].append(top_p)
            all_data['max_new_tokens'].append(max_new_tokens)
            all_data['tests'].append(tests)
            all_data['stop_tokens'].append(stop_tokens)
            all_data['program'].append(result['program'])
            all_data['timestamp'].append(result['timestamp'])
            all_data['stdout'].append(result['stdout'])
            all_data['stderr'].append(result['stderr'])
            all_data['exit_code'].append(result['exit_code'])
            all_data['status'].append(result['status'])
            all_data['model_name'].append(model_name)
    return all_data

def get_prompts_details(all_data):
    prompts_in_all_data = all_data['prompt_id'].unique().tolist()
    prompts_details = {
        'prompt_id': [],
        'prompt_number': [],
        'prompt': [],
        'Status_OK_count': [],
        'Status_SyntaxError_count': [],
        'Status_Timeout_count': [],
        'Status_Exception_count': [],
    }
    for current_prompt in prompts_in_all_data:
        prompt_df = all_data[all_data['prompt_id'] == current_prompt]
        prompt_number = prompt_df['prompt_number'].unique().tolist()[0]
        prompt = prompt_df['prompt'].unique().tolist()[0]
        Status_OK_count = prompt_df[prompt_df['status'] == 'OK'].shape[0]
        Status_SyntaxError_count = prompt_df[prompt_df['status'] == 'SyntaxError'].shape[0]
        Status_Timeout_count = prompt_df[prompt_df['status'] == 'Timeout'].shape[0]
        Status_Exception_count = prompt_df[prompt_df['status'] == 'Exception'].shape[0]
        prompts_details['prompt_id'].append(current_prompt)
        prompts_details['prompt_number'].append(prompt_number)
        prompts_details['prompt'].append(prompt)
        prompts_details['Status_OK_count'].append(Status_OK_count)
        prompts_details['Status_SyntaxError_count'].append(Status_SyntaxError_count)
        prompts_details['Status_Timeout_count'].append(Status_Timeout_count)
        prompts_details['Status_Exception_count'].append(Status_Exception_count)
    prompts_details_df = pd.DataFrame(prompts_details)
    return prompts_details_df
@st.cache_data
def all_flow(solution_dir, prompts_file, language=None):
    solutions = get_json_files(solution_dir)
    prompts = utils.read_prompts(prompts_file)
    prompts = pd.DataFrame(prompts)
    data = get_all_data(solutions, solution_dir, prompts)
    data_df = pd.DataFrame(data)
    prompts_details_df = get_prompts_details(data_df)
    if language is not None:
        prompts_details_df['language'] = language
    return data_df, prompts_details_df


def error_distribution(df):
    #Plot the distribution of errors
    #Set figure size
    fig = px.histogram(df, x='status', color='status', title='Error Distribution')
    st.write(fig)
    #wirtes the value counts for each error
    st.write(df['status'].value_counts().to_dict())

def solution_length_distribution(df):
    #Plot the distribution of solution lengths
    #Set figure size
    solutions = df['program'].tolist()
    solution_lengths = []
    for solution in solutions:
        solution_lengths.append(len(solution))
    fig = px.histogram(x=solution_lengths, title='Solution Length Distribution')
    st.write(fig)



def solution_details(df, key, prompt_number, number_of_prompts=100):
    models_names = df['model_name'].unique().tolist()
    models_names.insert(0, 'all')
    model_name = st.radio('Model Name', models_names, key=key*13)
    if model_name != 'all':
        df = df[df['model_name'] == model_name]

    st.write(f'Shape of Selected Dataframe: {df.shape}')
    st.write(f'Precentage of SyntaxError: {df[df["status"] == "SyntaxError"].shape[0] / df.shape[0] * 100:.2f}%')
    st.write(f'Precentage of Timeout: {df[df["status"] == "Timeout"].shape[0] / df.shape[0] * 100:.2f}%')
    st.write(f'Precentage of Exception: {df[df["status"] == "Exception"].shape[0] / df.shape[0] * 100:.2f}%')
    st.write(f'Precentage of OK: {df[df["status"] == "OK"].shape[0] / df.shape[0] * 100:.2f}%')
    
    error_distribution(df)
    solution_length_distribution(df)
    status_options = ['OK', 'SyntaxError', 'Timeout', 'Exception']
    status_options.insert(0, 'all')
    status = st.radio('Status', status_options, key=key*17)
    if status != 'all':
        df = df[df['status'] == status]

    df = df[df['prompt_number'] == prompt_number]
    df = df.reset_index(drop=True)

    st.write(df)

    st.write(df['status'].value_counts().to_dict())
    row_index = st.number_input('Row Index', 0, df.shape[0] - 1, 0, key=key*19)
    row = df.iloc[row_index]
    prompt_id = row['prompt_id']
    model_name = row['model_name']
    stderr = row['stderr']
    status = row['status']
    info_dict = {
        'prompt_id': prompt_id,
        'model_name': model_name,
        'stderr': stderr,
        'status': status,
    }
    st.write(info_dict)
    language = row['language']
    prompt = row['prompt']
    program = row['program']
    
    st.code(program, language=language, line_numbers=True)
    return df

def main():
    python_prompts_file = 'humaneval_py.jsonl'
    ruby_prompts_file = 'humaneval_rb.jsonl'
    rust_prompts_file = 'humaneval_rs.jsonl'
    swift_prompts_file = 'humaneval_swift.jsonl'
    java_prompts_file = 'humaneval_java.jsonl'
    python_solutions_dir = 'temp/tgt/codegen_350M_mono_humaneval_py'
    java_solutions_dir = 'temp/tgt/codegen_java_LoRa_java_pass_at_10'
    ruby_solutions_dir = 'temp/tgt/codegen_ruby_LoRa_rb_pass_at_10'
    rust_solutions_dir = 'temp/tgt/codegen_rust_LoRa_rs_pass_at_10'
    swift_solutions_dir = 'temp/tgt/codegen_swift_LoRa_swift_pass_at_10'
    

    python_data_df, python_prompts_details_df = all_flow(python_solutions_dir, python_prompts_file, 'python')
    java_data_df, java_prompts_details_df = all_flow(java_solutions_dir, java_prompts_file, 'java')
    ruby_data_df, ruby_prompts_details_df = all_flow(ruby_solutions_dir, ruby_prompts_file, 'ruby')
    rust_data_df, rust_prompts_details_df = all_flow(rust_solutions_dir, rust_prompts_file, 'rust')
    swift_data_df, swift_prompts_details_df = all_flow(swift_solutions_dir, swift_prompts_file, 'swift')


    prompts_details_df = pd.concat([python_prompts_details_df, java_prompts_details_df, ruby_prompts_details_df, rust_prompts_details_df, swift_prompts_details_df])
    st.write(prompts_details_df)

    #Create a line plot of of the number of each status for each prompt number for each language
    x_column = 'prompt_number'
    y_column = 'Status_OK_count'
    prompts_details_df = prompts_details_df.sort_values(by=['prompt_number'])

    fig = px.line(prompts_details_df, x=x_column, y=y_column, color='language', width=1800, height=800)

    #Add the length of each prompt as another line
    prompt_lengths = []
    for prompt in prompts_details_df['prompt']:
        prompt_lengths.append(len(prompt))
    #Normalize the prompt lengths to be bewteen 1 and 50
    prompt_lengths = np.array(prompt_lengths)
    prompt_lengths = (prompt_lengths - prompt_lengths.min()) / (prompt_lengths.max() - prompt_lengths.min())
    prompt_lengths = prompt_lengths * 49 + 1

    prompts_details_df['prompt_length'] = prompt_lengths
    fig.add_scatter(x=prompts_details_df[x_column], y=prompts_details_df['prompt_length'], mode='lines', name='Prompt Length')

    st.write(fig)
    

    #Combine the dataframes
    data_df = pd.concat([python_data_df, java_data_df, ruby_data_df, rust_data_df, swift_data_df])
    st.write(data_df)


    number_of_prompts = data_df['prompt_id'].unique().shape[0]
    # prompt_number = st.slider('Prompt Number', 1, number_of_prompts, 1, key=66)
    prompt_number = st.sidebar.number_input('Prompt Number', 1, number_of_prompts, 1, key=66)

    col1, col2 = st.columns(2)
    with col1:
        df_col1 = solution_details(data_df, 1,prompt_number, number_of_prompts)
        st.write(df_col1)
    with col2:
        df_col2 = solution_details(data_df, 2,prompt_number, number_of_prompts)
        st.write(df_col2)

    #Display value counts for each stderr
    # st.write(data_df['stderr'].value_counts().to_dict())

    #Display value counts for each status
    st.write(data_df['status'].value_counts().to_dict())

    #Number input for displaying a specific row
    row_index = st.number_input('Row Index', 0, data_df.shape[0] - 1, 0)

    #Display the row
    row = data_df.iloc[row_index]

    prompt_id = row['prompt_id']
    model_name = row['model_name']
    stderr = row['stderr']
    status = row['status']
    info_dict = {
        'prompt_id': prompt_id,
        'model_name': model_name,
        'stderr': stderr,
        'status': status,
    }
    st.write(info_dict)

    language = row['language']
    prompt = row['prompt']
    program = row['program']

    #Display the prompt
    st.code(program, language=language, line_numbers=True)

if __name__ == "__main__":
    # tab1, tab2 = st.tabs(["Code Generation", "Error Analysis"])
    # with tab1:
    #     code_generation.main()
    # with tab2:
    #     main()

    code_generation.main()