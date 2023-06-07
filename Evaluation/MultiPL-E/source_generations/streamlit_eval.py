import streamlit as st
import os
import subprocess

st.title("Code Generation Evaluation")

# User input for experiment name
exp_name = st.text_input("Experiment Name", "codegen-350M-mon_pass1_humaneval_py")

# User input for source file path
sf = st.text_input("Source File Path", f"{exp_name}.jsonl")

# Remove '-' from the experiment name
td = exp_name.replace("-", "")

# Button to trigger the conversion to pre-evaluation format
if st.button("Convert to Pre-evaluation"):
    # Run conversion command
    convert_cmd = f"python convert_to_pre_eval.py --source_file {sf} --target_dir {td}"
    subprocess.run(convert_cmd, shell=True)
    st.success("Conversion completed successfully!")

# Button to trigger multipl-e-eval execution
if st.button("Run multipl-e-eval"):
    # Run multipl-e-eval using podman
    eval_cmd = f"podman run --rm --network none -v ./{td}:/{td}:rw multipl-e-eval --dir /{td} --output-dir /{td} --recursive"
    subprocess.run(eval_cmd, shell=True)
    st.success("multipl-e-eval executed successfully!")

# Button to trigger pass_k.py execution
if st.button("Run pass_k.py"):
    # Change directory to parent directory
    os.chdir("..")
    
    # Run pass_k.py
    target_dir = f"source_generations/{td}"
    output_file = f"source_generations/{exp_name}_results.json"
    pass_k_cmd = f"python pass_k.py {target_dir} {output_file}"
    subprocess.run(pass_k_cmd, shell=True)
    st.success("pass_k.py executed successfully!")

    # Display the output file link
    output_link = f"[Download {output_file}](/{output_file})"
    st.markdown(output_link, unsafe_allow_html=True)
