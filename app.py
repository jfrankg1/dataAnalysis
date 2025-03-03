import streamlit as st
import pandas as pd
import numpy as np
import anthropic
import os
from io import StringIO
import base64

# Page configuration
st.set_page_config(page_title="Assay Data Normalizer", layout="wide")

def main():
    st.title("Experimental Assay Data Normalizer")
    st.write("Upload your CSV files, select normalization method, and get processed results with AI-assisted analysis.")
    
    # API key input (with warning about security)
    with st.expander("Claude API Configuration"):
        st.warning("Note: For production use, you should use environment variables or secure methods to store API keys.")
        api_key = st.secrets["anthropic"]["api_key"]
    
    # File uploader for multiple CSV files
    uploaded_files = st.file_uploader("Upload CSV file(s)", type="csv", accept_multiple_files=True)
    
    if not uploaded_files:
        st.info("Please upload one or more CSV files to begin.")
        return
    
    # Normalization method selection
    norm_method = st.selectbox(
        "Select normalization method:",
        ["None", "Z-score", "Percent activity"]
    )
    
    # Process button
    if st.button("Process Data"):
        if not api_key and norm_method != "None":
            st.error("Please enter your Anthropic API key to use AI-assisted normalization.")
            return
            
        processed_dfs = []
        for uploaded_file in uploaded_files:
            # Process each file
            with st.spinner(f"Processing {uploaded_file.name}..."):
                df = process_file(uploaded_file, norm_method, api_key)
                processed_dfs.append((uploaded_file.name, df))
        
        # Display and provide download links for processed files
        if processed_dfs:
            st.success("Processing complete! Results are available below.")
            display_results(processed_dfs)

def process_file(file, norm_method, api_key):
    # Read the CSV file
    df = pd.read_csv(file)
    
    # Display original data
    st.subheader(f"Original Data: {file.name}")
    st.dataframe(df.head())
    
    # Identify controls if needed (using Claude)
    control_info = None
    if norm_method in ["Z-score", "Percent activity"]:
        control_info = identify_controls(df, api_key)
    
    # Apply normalization
    if norm_method == "None":
        # No normalization needed
        normalized_df = df.copy()
        st.write("No normalization applied.")
    else:
        # Apply selected normalization
        normalized_df = normalize_data(df, norm_method, control_info)
    
    return normalized_df

def identify_controls(df, api_key):
    """Use Claude to identify positive and negative controls in the dataset"""
    st.write("Analyzing data to identify controls...")
    
    # Create a sample of the data to send to Claude
    data_sample = df.head(10).to_string()
    
    # Get column names
    columns = df.columns.tolist()
    
    prompt = f"""
    I have an experimental assay dataset with the following columns:
    {columns}
    
    Here's a sample of the data:
    {data_sample}
    
    Based on this information, please:
    1. Identify which columns might contain positive and negative controls
    2. Suggest how to determine which rows or values are positive or negative controls
    3. Return your answer in a structured JSON format with 'positive_controls' and 'negative_controls' fields
    
    Be specific about how you would identify the controls from the data structure.
    """
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1000,
            temperature=0,
            system="You are a helpful AI assistant with expertise in biological assay data analysis.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        response = message.content[0].text
        st.write("Claude analysis:")
        st.write(response)
        
        # Note: In a production app, you'd parse the JSON response
        # For demo purposes, we'll use a placeholder
        control_info = {
            "positive_controls": "Based on Claude's recommendation",
            "negative_controls": "Based on Claude's recommendation"
        }
        return control_info
        
    except Exception as e:
        st.error(f"Error connecting to Claude API: {str(e)}")
        return None

def normalize_data(df, method, control_info):
    """Apply the selected normalization method to the data"""
    result_df = df.copy()
    
    # For demonstration, we'll normalize all numeric columns
    # In a real app, you'd use the control_info to be more selective
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if method == "Z-score":
        st.write("Applying Z-score normalization...")
        for col in numeric_cols:
            mean = df[col].mean()
            std = df[col].std()
            if std != 0:  # Avoid division by zero
                result_df[col] = (df[col] - mean) / std
            else:
                st.warning(f"Column {col} has zero standard deviation, skipping normalization.")
    
    elif method == "Percent activity":
        st.write("Applying percent activity normalization...")
        for col in numeric_cols:
            # In a real app, you would use positive and negative controls from control_info
            # For this demo, we'll use min/max as proxies
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val - min_val != 0:  # Avoid division by zero
                result_df[col] = ((df[col] - min_val) / (max_val - min_val)) * 100
            else:
                st.warning(f"Column {col} has identical min/max values, skipping normalization.")
    
    return result_df

def display_results(processed_dfs):
    """Display results and provide download links"""
    for filename, df in processed_dfs:
        st.subheader(f"Processed Data: {filename}")
        st.dataframe(df.head(10))
        
        # Create download link
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        processed_filename = f"normalized_{filename}"
        href = f'<a href="data:file/csv;base64,{b64}" download="{processed_filename}">Download processed file</a>'
        st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
