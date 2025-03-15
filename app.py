import streamlit as st
import pandas as pd
import numpy as np
import anthropic
import json
import re
import os
from io import StringIO
import base64
from typing import Dict, List, Tuple, Any, Optional

# Page configuration
st.set_page_config(
    page_title="Experimental Assay Data Processor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for storing processed data
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = {}

def main():
    st.title("Experimental Assay Data Processor")
    st.write("Upload your experimental assay data files for AI-assisted analysis.")
    
    # API key configuration
    with st.sidebar:
        st.header("Claude API Configuration")
        api_key_source = st.radio(
            "Claude API Key Source",
            options=["Use from secrets", "Enter manually"],
            index=0
        )
        
        if api_key_source == "Use from secrets":
            try:
                api_key = st.secrets["anthropic"]["api_key"]
                st.success("API key loaded from secrets")
            except Exception as e:
                st.error(f"Failed to load API key from secrets: {e}")
                api_key = None
        else:
            api_key = st.text_input("Enter Claude API Key", type="password")
            
        if not api_key:
            st.warning("API key not configured - AI-assisted analysis will not be available")
    
    # File uploader for multiple CSV files
    uploaded_files = st.file_uploader(
        "Upload CSV file(s) with experimental assay data", 
        type=["csv", "txt"], 
        accept_multiple_files=True
    )
    
    if not uploaded_files:
        st.info("Please upload one or more CSV files to begin.")
        st.stop()
    
    # Display file summary
    st.subheader("Uploaded Files")
    for idx, file in enumerate(uploaded_files):
        st.write(f"{idx+1}. {file.name}")
    
    # Process button
    if st.button("Process Data", type="primary"):
        if not api_key:
            st.error("Please configure your Claude API key to use AI-assisted analysis.")
            st.stop()
        
        # Initialize processed files dictionary
        processed_files = {}
        
        # Create progress bar
        progress_bar = st.progress(0)
        
        # Process each file individually
        for file_idx, file in enumerate(uploaded_files):
            file_id = f"file_{file_idx}"
            
            with st.expander(f"Processing {file.name}", expanded=True):
                st.markdown(f"#### Analyzing {file.name}")
                
                # Initialize the data structure for this file
                processed_files[file_id] = {
                    "metadata": {
                        "filename": file.name,
                        "original_shape": None,
                        "orientation": None,
                        "header_location": None,
                        "sample_id_location": None
                    },
                    "data": {
                        "raw_df": None,
                        "processed_df": None,
                        "headers": [],
                        "sample_ids": [],
                        "controls": {
                            "positive": {
                                "sample_ids": [],
                                "indices": [],
                                "data": None
                            },
                            "negative": {
                                "sample_ids": [],
                                "indices": [],
                                "data": None
                            }
                        }
                    },
                    "analysis_results": {
                        "z_scores": None,
                        "percent_activity": None
                    }
                }
                
                # Step 1: Read the file and store the raw data
                file_data = process_file(file, api_key)
                processed_files[file_id] = file_data
                
                # Step 2: Display extracted information
                st.markdown("### File Structure")
                st.write(f"- Orientation: {file_data['metadata']['orientation']}")
                st.write(f"- Header location: {file_data['metadata']['header_location']['type']} index {file_data['metadata']['header_location']['index']}")
                st.write(f"- Sample ID location: {file_data['metadata']['sample_id_location']['type']} index {file_data['metadata']['sample_id_location']['index']}")
                
                # Display headers
                st.markdown("### Data Headers")
                headers = get_headers(file_data)
                st.write(headers)
                
                # Display sample IDs
                st.markdown("### Sample IDs")
                sample_ids = get_sample_ids(file_data)
                st.write(sample_ids)
                
                # Display controls if identified
                if file_data["data"]["controls"]["positive"]["data"] is not None:
                    st.markdown("### Positive Controls")
                    pos_controls = get_positive_controls(file_data)
                    st.dataframe(pos_controls)
                else:
                    st.warning("No positive controls identified.")
                
                if file_data["data"]["controls"]["negative"]["data"] is not None:
                    st.markdown("### Negative Controls")
                    neg_controls = get_negative_controls(file_data)
                    st.dataframe(neg_controls)
                else:
                    st.warning("No negative controls identified.")
                
                st.success(f"Successfully processed {file.name}")
            
            # Update progress
            progress_bar.progress((file_idx + 1) / len(uploaded_files))
        
        # Store in session state for use in other modules
        st.session_state.processed_files = processed_files
        
        # Complete
        progress_bar.progress(100)
        st.success("All files processed successfully!")

def process_file(file, api_key) -> Dict:
    """
    Process an uploaded CSV file using Claude 3.7 Sonnet to identify structure and controls.
    
    Args:
        file: The uploaded file object
        api_key: The Claude API key
        
    Returns:
        Dict: A dictionary with the processed file data in the specified structure
    """
    # Initialize the data structure
    file_data = {
        "metadata": {
            "filename": file.name,
            "original_shape": None,
            "orientation": None,
            "header_location": None,
            "sample_id_location": None
        },
        "data": {
            "raw_df": None,
            "processed_df": None,
            "headers": [],
            "sample_ids": [],
            "controls": {
                "positive": {
                    "sample_ids": [],
                    "indices": [],
                    "data": None
                },
                "negative": {
                    "sample_ids": [],
                    "indices": [],
                    "data": None
                }
            }
        },
        "analysis_results": {
            "z_scores": None,
            "percent_activity": None
        }
    }
    
    # Read the file into a pandas DataFrame
    with st.spinner("Reading file..."):
        file.seek(0)
        try:
            # Try to read with default parameters first
            raw_df = pd.read_csv(file)
        except:
            # If that fails, try with more flexible parameters
            file.seek(0)
            raw_df = pd.read_csv(file, sep=None, engine='python')
        
        # Store the raw data
        file_data["data"]["raw_df"] = raw_df
        file_data["metadata"]["original_shape"] = raw_df.shape
    
    # Analyze file structure with Claude
    with st.spinner("Analyzing file structure with Claude 3.7 Sonnet..."):
        structure_info = analyze_file_structure(raw_df, api_key)
        
        # Update metadata based on analysis
        file_data["metadata"]["orientation"] = structure_info.get("orientation", "standard")
        file_data["metadata"]["header_location"] = structure_info.get("header_location", {"type": "row", "index": 0})
        file_data["metadata"]["sample_id_location"] = structure_info.get("sample_id_location", {"type": "column", "index": 0})
        
        # Process the dataframe based on the identified structure
        processed_df, headers, sample_ids = process_dataframe(
            raw_df, 
            structure_info["orientation"],
            structure_info["header_location"],
            structure_info["sample_id_location"]
        )
        
        file_data["data"]["processed_df"] = processed_df
        file_data["data"]["headers"] = headers
        file_data["data"]["sample_ids"] = sample_ids
    
    # Identify controls with Claude
    with st.spinner("Identifying control samples..."):
        control_info = identify_controls(processed_df, headers, sample_ids, api_key)
        
        # Process positive controls
        if control_info.get("positive", {}).get("sample_ids"):
            # Save the original sample IDs from Claude
            all_pos_ids = control_info["positive"]["sample_ids"]
            
            # Filter out any non-string values or NaN values
            valid_pos_ids = []
            for id in all_pos_ids:
                if id is None or pd.isna(id):
                    continue
                valid_pos_ids.append(str(id))
            
            # Only keep IDs that exist in sample_ids
            valid_pos_ids = [id for id in valid_pos_ids if id in sample_ids]
            
            # Store valid IDs
            file_data["data"]["controls"]["positive"]["sample_ids"] = valid_pos_ids
            
            # Safely convert to indices
            try:
                pos_indices = [sample_ids.index(id) for id in valid_pos_ids]
                file_data["data"]["controls"]["positive"]["indices"] = pos_indices
                
                # Extract positive control data only if we have valid indices
                if pos_indices:
                    file_data["data"]["controls"]["positive"]["data"] = processed_df.iloc[pos_indices]
                else:
                    st.warning("No valid positive control indices found after filtering")
            except Exception as e:
                st.warning(f"Error extracting positive control data: {e}")
        
        # Process negative controls
        if control_info.get("negative", {}).get("sample_ids"):
            # Save the original sample IDs from Claude
            all_neg_ids = control_info["negative"]["sample_ids"]
            
            # Filter out any non-string values or NaN values
            valid_neg_ids = []
            for id in all_neg_ids:
                if id is None or pd.isna(id):
                    continue
                valid_neg_ids.append(str(id))
            
            # Only keep IDs that exist in sample_ids
            valid_neg_ids = [id for id in valid_neg_ids if id in sample_ids]
            
            # Store valid IDs
            file_data["data"]["controls"]["negative"]["sample_ids"] = valid_neg_ids
            
            # Safely convert to indices
            try:
                neg_indices = [sample_ids.index(id) for id in valid_neg_ids]
                file_data["data"]["controls"]["negative"]["indices"] = neg_indices
                
                # Extract negative control data only if we have valid indices
                if neg_indices:
                    file_data["data"]["controls"]["negative"]["data"] = processed_df.iloc[neg_indices]
                else:
                    st.warning("No valid negative control indices found after filtering")
            except Exception as e:
                st.warning(f"Error extracting negative control data: {e}")
    
    return file_data

def analyze_file_structure(df, api_key) -> Dict:
    """
    Use Claude 3.7 Sonnet to analyze the structure of the file.
    
    Args:
        df: The pandas DataFrame containing the raw data
        api_key: The Claude API key
        
    Returns:
        Dict: A dictionary with the structure information
    """
    # Create sample of the data to send to Claude
    # Limit to first 10 rows and columns to avoid token limits
    max_rows = min(10, df.shape[0])
    max_cols = min(10, df.shape[1])
    
    data_sample = df.iloc[:max_rows, :max_cols].to_string()
    
    # Create the client
    client = anthropic.Anthropic(api_key=api_key)
    
    # Prepare the prompt
    prompt = f"""
    I have a scientific dataset from an experimental assay in CSV format.
    I need to determine:
    
    1. If the data is in standard or transposed orientation
        - Standard: Samples in rows, measurements in columns
        - Transposed: Samples in columns, measurements in rows
    
    2. The location of data headers (describing what each measurement is)
        - In standard orientation: Which row contains headers
        - In transposed orientation: Which column contains headers
    
    3. The location of sample IDs
        - In standard orientation: Which column contains sample IDs
        - In transposed orientation: Which row contains sample IDs
    
    Here's a sample of the data:
    ```
    {data_sample}
    ```
    
    Use your expertise in scientific data analysis to determine the file structure.
    Return your analysis in JSON format:
    
    {{
        "orientation": "standard" or "transposed",
        "header_location": {{"type": "row" or "column", "index": integer}},
        "sample_id_location": {{"type": "row" or "column", "index": integer}},
        "reasoning": "brief explanation of your analysis"
    }}
    """
    
    try:
        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1000,
            temperature=0,
            system="You are an expert in scientific data analysis. Answer with JSON only.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        response = message.content[0].text
        
        # Extract JSON from response
        try:
            json_match = re.search(r'\{.*\}', response.replace('\n', ''), re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                st.info(f"Structure analysis: {result.get('reasoning', 'No reasoning provided')}")
                return result
            else:
                st.warning("Could not parse Claude's response. Using default structure.")
                return {
                    "orientation": "standard",
                    "header_location": {"type": "row", "index": 0},
                    "sample_id_location": {"type": "column", "index": 0}
                }
        except json.JSONDecodeError:
            st.warning("Invalid JSON in Claude's response. Using default structure.")
            return {
                "orientation": "standard",
                "header_location": {"type": "row", "index": 0},
                "sample_id_location": {"type": "column", "index": 0}
            }
            
    except Exception as e:
        st.error(f"Error using Claude API: {str(e)}")
        return {
            "orientation": "standard",
            "header_location": {"type": "row", "index": 0},
            "sample_id_location": {"type": "column", "index": 0}
        }

def process_dataframe(df, orientation, header_location, sample_id_location) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Process the dataframe based on the identified structure.
    
    Args:
        df: The pandas DataFrame to process
        orientation: "standard" or "transposed"
        header_location: Dict with "type" and "index"
        sample_id_location: Dict with "type" and "index"
        
    Returns:
        Tuple: (processed_df, headers, sample_ids)
    """
    # Initialize
    processed_df = df.copy()
    headers = []
    sample_ids = []
    
    # Step 1: Extract headers based on location
    if orientation == "standard" and header_location["type"] == "row":
        # Headers are in a row
        header_idx = header_location["index"]
        headers = df.iloc[header_idx].tolist()
        
        # Filter out NaN values from headers
        headers = [str(h) for h in headers if not pd.isna(h)]
        
        # If header is not the first row, reset the dataframe
        if header_idx > 0:
            processed_df = df.iloc[header_idx:].copy()
            processed_df.columns = processed_df.iloc[0]
            processed_df = processed_df.iloc[1:]
            processed_df.reset_index(drop=True, inplace=True)
    
    elif orientation == "transposed" and header_location["type"] == "column":
        # Headers are in a column
        header_idx = header_location["index"]
        headers = df.iloc[:, header_idx].tolist()
        
        # Filter out NaN values from headers
        headers = [str(h) for h in headers if not pd.isna(h)]
        
        # Transpose the dataframe so samples are in rows
        processed_df = df.transpose()
        
        # If header is not the first column, handle accordingly
        if header_idx > 0:
            # Use the specified column as header after transposing
            processed_df.columns = processed_df.iloc[header_idx]
            processed_df = processed_df.iloc[1:]
            processed_df.reset_index(drop=True, inplace=True)
    
    # Step 2: Extract sample IDs based on location
    if orientation == "standard" and sample_id_location["type"] == "column":
        # Sample IDs are in a column
        id_idx = sample_id_location["index"]
        sample_ids = df.iloc[:, id_idx].tolist()
        
        # Filter out NaN values from sample IDs
        sample_ids = [str(s) for s in sample_ids if not pd.isna(s)]
        
        # Exclude header row if it exists
        if header_location["type"] == "row" and header_location["index"] <= len(sample_ids) - 1:
            sample_ids = sample_ids[header_location["index"]+1:]
    
    elif orientation == "transposed" and sample_id_location["type"] == "row":
        # Sample IDs are in a row
        id_idx = sample_id_location["index"]
        sample_ids = df.iloc[id_idx].tolist()
        
        # Filter out NaN values from sample IDs
        sample_ids = [str(s) for s in sample_ids if not pd.isna(s)]
        
        # Account for transposition
        if header_location["type"] == "column" and header_location["index"] <= len(sample_ids) - 1:
            sample_ids = sample_ids[header_location["index"]+1:]
    
    return processed_df, headers, sample_ids

def identify_controls(df, headers, sample_ids, api_key) -> Dict:
    """
    Use Claude 3.7 Sonnet to identify positive and negative control samples.
    Analyzes the entire dataset for more accurate control identification.
    
    Args:
        df: The processed pandas DataFrame
        headers: List of data headers
        sample_ids: List of sample IDs
        api_key: The Claude API key
        
    Returns:
        Dict: A dictionary with positive and negative control information
    """
    # Create a comprehensive data representation for Claude
    sample_id_str = "\n".join([f"{i}: {id}" for i, id in enumerate(sample_ids)])
    
    # Get the shape of the dataframe
    rows, cols = df.shape
    
    # Create a token-efficient but thorough data representation
    data_info = []
    data_info.append(f"Dataset has {rows} rows and {cols} columns.")
    data_info.append(f"Headers: {headers}")
    
    # Get numeric columns for analysis
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Add statistical summary for better context
    if numeric_cols:
        data_info.append("\nStatistical summary of numeric columns:")
        stats = df[numeric_cols].describe().to_string()
        data_info.append(stats)
    
    # Include the entire dataset, handling large datasets appropriately
    if rows > 100:
        # For large datasets, include full sample ID list but sample the data
        data_info.append(f"\nFull data (first 20 rows shown as sample):")
        data_info.append(df.head(20).to_string())
        # Add a sample from the middle and end
        if rows > 50:
            middle_idx = rows // 2
            data_info.append(f"\nData from middle of dataset (rows {middle_idx}-{middle_idx+4}):")
            data_info.append(df.iloc[middle_idx:middle_idx+5].to_string())
            data_info.append(f"\nData from end of dataset (last 5 rows):")
            data_info.append(df.tail(5).to_string())
    else:
        # For smaller datasets, include the entire dataset
        data_info.append("\nFull dataset:")
        data_info.append(df.to_string())
    
    # Check for patterns in sample IDs that might indicate controls
    sample_id_patterns = "\nSample ID analysis:"
    pos_patterns = ["positive", "pos", "high", "max", "+", "pos_ctrl", "pos ctrl"]
    neg_patterns = ["negative", "neg", "low", "min", "-", "blank", "buffer", "neg_ctrl", "neg ctrl"]
    
    potential_pos = []
    potential_neg = []
    
    for i, sample_id in enumerate(sample_ids):
        sid_lower = str(sample_id).lower()
        
        is_pos = any(pattern in sid_lower for pattern in pos_patterns)
        is_neg = any(pattern in sid_lower for pattern in neg_patterns)
        
        if is_pos:
            potential_pos.append(i)
        if is_neg:
            potential_neg.append(i)
    
    if potential_pos:
        sample_id_patterns += f"\nPotential positive controls based on names: {[sample_ids[i] for i in potential_pos]}"
        # Include data values for these potential controls
        for i in potential_pos:
            if i < len(df) and i < rows:
                row_data = df.iloc[i]
                sample_id_patterns += f"\n{sample_ids[i]} values: "
                for col in numeric_cols[:5]:  # Limit columns to save tokens
                    if col in df.columns:
                        val = row_data[col]
                        sample_id_patterns += f"{col}={val}, "
    
    if potential_neg:
        sample_id_patterns += f"\nPotential negative controls based on names: {[sample_ids[i] for i in potential_neg]}"
        # Include data values for these potential controls
        for i in potential_neg:
            if i < len(df) and i < rows:
                row_data = df.iloc[i]
                sample_id_patterns += f"\n{sample_ids[i]} values: "
                for col in numeric_cols[:5]:  # Limit columns to save tokens
                    if col in df.columns:
                        val = row_data[col]
                        sample_id_patterns += f"{col}={val}, "
    
    data_info.append(sample_id_patterns)
    
    # Look for extreme values that might indicate controls
    if numeric_cols:
        extreme_values = "\nRows with extreme values that might be controls:"
        
        for col in numeric_cols[:5]:  # Limit to first 5 numeric columns to save tokens
            if col in df.columns:
                # Find rows with max values
                max_indices = df[col].nlargest(3).index.tolist()
                extreme_values += f"\nHighest {col} values: "
                for idx in max_indices:
                    if idx < len(sample_ids):
                        sample_name = sample_ids[idx] if idx < len(sample_ids) else f"Row {idx}"
                        extreme_values += f"{sample_name}={df.iloc[idx][col]}, "
                
                # Find rows with min values
                min_indices = df[col].nsmallest(3).index.tolist()
                extreme_values += f"\nLowest {col} values: "
                for idx in min_indices:
                    if idx < len(sample_ids):
                        sample_name = sample_ids[idx] if idx < len(sample_ids) else f"Row {idx}"
                        extreme_values += f"{sample_name}={df.iloc[idx][col]}, "
        
        data_info.append(extreme_values)
    
    # Join all data information
    data_summary = "\n".join(data_info)
    
    # Show warning for large datasets
    if rows * cols > 5000:
        st.warning("Analyzing a large dataset. This might take longer than usual.")
    
    # Create the client
    client = anthropic.Anthropic(api_key=api_key)
    
    # Prepare the prompt
    prompt = f"""
    I have an experimental assay dataset and need to identify the positive and negative control samples.
    
    Here are the sample IDs:
    {sample_id_str}
    
    And here's comprehensive information about the dataset:
    ```
    {data_summary}
    ```
    
    In scientific assays, controls typically have these characteristics:
    
    1. Positive controls:
       - Often labeled with terms like "positive", "pos", "pos ctrl", "high", "max", etc.
       - Usually have higher measurement values than other samples
       - May be standard compounds with known high activity
    
    2. Negative controls:
       - Often labeled with terms like "negative", "neg", "neg ctrl", "low", "min", "blank", "buffer", etc.
       - Usually have lower measurement values than other samples
       - May be buffer-only or vehicle-only samples
    
    Based on the COMPLETE dataset information I've provided (including patterns in names, extreme values, 
    and statistical distributions), identify which samples are positive and negative controls.
    
    Return your analysis in JSON format:
    
    {{
        "positive": {{
            "sample_ids": [list of sample IDs that are positive controls],
            "reasoning": "brief explanation"
        }},
        "negative": {{
            "sample_ids": [list of sample IDs that are negative controls],
            "reasoning": "brief explanation"
        }}
    }}
    """
    
    try:
        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1000,
            temperature=0,
            system="You are an expert in scientific data analysis. Answer with JSON only.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        response = message.content[0].text
        
        # Extract JSON from response
        try:
            json_match = re.search(r'\{.*\}', response.replace('\n', ''), re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                
                # Log the reasoning
                pos_reasoning = result.get("positive", {}).get("reasoning", "No reasoning provided")
                neg_reasoning = result.get("negative", {}).get("reasoning", "No reasoning provided")
                
                st.info(f"Positive controls: {pos_reasoning}")
                st.info(f"Negative controls: {neg_reasoning}")
                
                return result
            else:
                st.warning("Could not parse Claude's response for control identification.")
                return {"positive": {"sample_ids": []}, "negative": {"sample_ids": []}}
        except json.JSONDecodeError:
            st.warning("Invalid JSON in Claude's response for control identification.")
            return {"positive": {"sample_ids": []}, "negative": {"sample_ids": []}}
            
    except Exception as e:
        st.error(f"Error using Claude API for control identification: {str(e)}")
        return {"positive": {"sample_ids": []}, "negative": {"sample_ids": []}}

# Helper functions provided in the prompt
def get_headers(file_data):
    """Extract just the data headers from a processed file"""
    return file_data["data"]["headers"]

def get_sample_ids(file_data):
    """Extract just the sample IDs from a processed file"""
    return file_data["data"]["sample_ids"]

def get_positive_controls(file_data):
    """Extract positive control data with headers"""
    return file_data["data"]["controls"]["positive"]["data"]

def get_negative_controls(file_data):
    """Extract negative control data with headers"""
    return file_data["data"]["controls"]["negative"]["data"]

if __name__ == "__main__":
    main()
