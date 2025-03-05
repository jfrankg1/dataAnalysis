import streamlit as st
import pandas as pd
import numpy as np
import anthropic
import os
from io import StringIO
import base64
import json
import re
from typing import List, Dict, Tuple, Optional, Any
import tiktoken

# Page configuration
st.set_page_config(
    page_title="Experimental Assay Data Processor", 
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("Experimental Assay Data Processor")
    st.write("Upload your experimental assay data files, select normalization method, and get AI-assisted analysis.")
    
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
                st.error("Failed to load API key from secrets")
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
    
    # Normalization method selection
    norm_method = st.selectbox(
        "Select normalization method:",
        ["None", "Z-score", "Percent activity"],
        help="Z-score normalizes data to have mean 0 and standard deviation 1. " +
             "Percent activity calculates values as a percentage between negative and positive controls."
    )
    
    # Process button
    if st.button("Process Data", type="primary"):
        if norm_method != "None" and not api_key:
            st.error("Please configure your Claude API key to use AI-assisted normalization.")
            st.stop()
        
        # Process each file individually
        processed_files = []
        intermediate_files = []
        
        for file in uploaded_files:
            st.markdown(f"### Processing {file.name}")
            
            # Step 1-4: Analyze data structure and orientation
            with st.spinner(f"Analyzing data structure in {file.name}..."):
                df, oriented_correctly, has_labels = analyze_file_structure(file, api_key)
                
                if not has_labels:
                    st.error(f"Could not identify data labels in {file.name}. Analysis stopped.")
                    continue
                
                if not oriented_correctly:
                    st.info(f"Data in {file.name} has been transposed - labels were found in columns.")
                    # Save intermediate file for user reference
                    intermediate_files.append((f"transposed_{file.name}", df))
            
            # Step 5: Review sample of data to glean additional information
            with st.spinner(f"Analyzing data samples in {file.name}..."):
                data_insights = analyze_data_sample(df, api_key)
                if data_insights:
                    with st.expander("Data insights"):
                        st.write(data_insights)
            
            # Step 6: Identify and validate controls
            with st.spinner(f"Identifying controls in {file.name}..."):
                controls, control_stats = identify_and_validate_controls(df, api_key)
                
                if not controls or not controls.get("positive_controls") or not controls.get("negative_controls"):
                    st.warning(f"Could not identify sufficient controls in {file.name}. Normalization may be limited.")
                
                # Display control information
                if controls:
                    st.write("Controls identified:")
                    pos_count = len(controls.get("positive_controls", []))
                    neg_count = len(controls.get("negative_controls", []))
                    st.write(f"- Positive controls: {pos_count}")
                    st.write(f"- Negative controls: {neg_count}")
                    
                    # Show control stats if available
                    if control_stats:
                        with st.expander("Control Statistics"):
                            st.json(control_stats)
            
            # Step 7: Normalize the data
            with st.spinner(f"Normalizing data in {file.name}..."):
                normalized_df = normalize_data(df, norm_method, controls, control_stats)
                processed_files.append((file.name, normalized_df, {
                    "controls": controls,
                    "control_stats": control_stats
                }))
                
                st.success(f"Successfully processed {file.name}")
        
        # Collective analysis across all files
        if len(processed_files) > 1:
            st.markdown("### Analyzing across all files")
            
            # Step 1: Unify data labels
            with st.spinner("Unifying data labels across files..."):
                unified_labels = unify_data_labels([df for _, df, _ in processed_files])
                st.write(f"Identified {len(unified_labels)} common data labels across files")
            
            # Step 2-4: Check for samples across multiple files and join if needed
            with st.spinner("Checking for samples in multiple files..."):
                combined_df = combine_samples_across_files(processed_files)
                if combined_df is not None:
                    st.success("Created combined output with samples from multiple files")
                    processed_files.append(("combined_samples.csv", combined_df, {
                        "is_combined": True
                    }))
        
        # Display results for download
        st.markdown("### Results")
        
        # Show intermediate files if any were created
        if intermediate_files:
            with st.expander("Intermediate Files (Transposed Data)"):
                for filename, df in intermediate_files:
                    st.write(f"**{filename}**")
                    st.dataframe(df.head())
                    csv = df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download intermediate file</a>'
                    st.markdown(href, unsafe_allow_html=True)
        
        # Show processed files
        for filename, df, metadata in processed_files:
            with st.expander(f"Processed: {filename}"):
                st.dataframe(df.head())
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                processed_filename = f"normalized_{filename}"
                href = f'<a href="data:file/csv;base64,{b64}" download="{processed_filename}">Download processed file</a>'
                st.markdown(href, unsafe_allow_html=True)

def analyze_file_structure(file, api_key) -> Tuple[pd.DataFrame, bool, bool]:
    """
    Analyzes the structure of a CSV file to determine:
    1. If it has data labels
    2. If data labels are in rows or columns
    
    Returns:
    - DataFrame (possibly transposed)
    - Whether it's already correctly oriented (True) or needed transposition (False)
    - Whether data labels were found (True/False)
    """
    # Reset file pointer and read file
    file.seek(0)
    df = pd.read_csv(file, skip_blank_lines=True)
    
    # Create sample of first row and first column for analysis
    first_row = df.iloc[0].tolist()
    first_col = df.iloc[:, 0].tolist()
    
    # Check if the first row looks like it contains labels
    row_has_labels = True
    col_has_labels = True
    
    # Simple heuristics for initial check
    numeric_in_first_row = sum(1 for x in first_row if isinstance(x, (int, float)) and not pd.isna(x))
    numeric_in_first_col = sum(1 for x in first_col if isinstance(x, (int, float)) and not pd.isna(x))
    
    # If first row is mostly numeric, it probably doesn't contain labels
    if numeric_in_first_row / len(first_row) > 0.5:
        row_has_labels = False
    
    # If first col is mostly numeric, it probably doesn't contain labels
    if numeric_in_first_col / len(first_col) > 0.5:
        col_has_labels = False
    
    # If we have API key, ask Claude to make a judgment call
    if api_key:
        # Estimate token count for first row and column content
        row_string = str(first_row)
        col_string = str(first_col)
        row_tokens = len(row_string.split())
        col_tokens = len(col_string.split())
        
        if row_tokens > 1000 or col_tokens > 1000:
            st.warning("Data sample is large and may exceed token limits for AI analysis.")
            proceed = st.checkbox("Proceed with sending large data sample to AI?", value=False)
            if not proceed:
                # Fall back to heuristics
                if row_has_labels:
                    return df, True, True
                elif col_has_labels:
                    return df.T, False, True
                else:
                    return df, True, False
        
        client = anthropic.Anthropic(api_key=api_key)
        
        # Prepare sample data
        sample_data = f"First row: {first_row}\n\nFirst column: {first_col}"
        
        prompt = f"""
        I have a scientific dataset from an experimental assay. I need to determine if:
        1. The first row contains data labels/headers
        2. The first column contains data labels/headers
        3. Neither contains labels (just data values)
        
        Here's the data:
        {sample_data}
        
        In scientific assay data, labels typically contain text descriptors like "Sample", "Control", 
        measurement names like "Concentration", "Absorbance", or sample identifiers.
        
        Raw data values are typically numerical, except for sample identifiers.
        
        Based on this data, please analyze:
        - Does the first row look like it contains labels/headers?
        - Does the first column look like it contains labels/headers?
        
        Reply with JSON only, in this format:
        {{
            "row_has_labels": true/false,
            "col_has_labels": true/false,
            "reasoning": "brief explanation"
        }}
        """
        
        try:
            message = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=500,
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
                    row_has_labels = result.get("row_has_labels", row_has_labels)
                    col_has_labels = result.get("col_has_labels", col_has_labels)
                    reasoning = result.get("reasoning", "")
                    st.info(f"AI analysis: {reasoning}")
            except:
                st.warning("Could not parse AI response. Using heuristic analysis instead.")
        except Exception as e:
            st.warning(f"Could not use AI for structure analysis: {str(e)}. Using heuristic analysis instead.")
    
    # Determine if we need to transpose and if we have labels
    if row_has_labels:
        return df, True, True
    elif col_has_labels:
        return df.T, False, True
    else:
        return df, True, False

def analyze_data_sample(df, api_key):
    """
    Review a sample of the data to glean additional information.
    Returns insights as text.
    """
    if not api_key:
        return None
    
    try:
        # Create a sample of the data
        data_sample = df.head(10).to_string()
        
        client = anthropic.Anthropic(api_key=api_key)
        prompt = f"""
        I have a sample of experimental assay data. Please analyze this sample and provide insights 
        about the data structure, potential patterns, or any information that might be relevant for analysis.
        
        Sample data:
        {data_sample}
        
        Please provide brief insights about:
        1. The apparent type of assay or experiment
        2. Key variables or measurements present
        3. Any patterns or anomalies in the data
        4. Any other relevant observations
        
        Keep your response brief and focused on information that would help with data normalization.
        """
        
        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=500,
            temperature=0,
            system="You are a data scientist specializing in experimental biology.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return message.content[0].text
        
    except Exception as e:
        st.warning(f"Could not get data insights: {str(e)}")
        return None

def identify_and_validate_controls(df, api_key):
    """
    Identify positive and negative controls in the dataset and validate their quality.
    """
    controls = {
        "positive_controls": [],
        "negative_controls": []
    }
    
    control_stats = {
        "positive": {},
        "negative": {}
    }
    
    # Simple heuristic approach first
    # Look for columns that might indicate control type
    control_indicators = []
    for col in df.columns:
        col_lower = str(col).lower()
        if 'type' in col_lower or 'control' in col_lower or 'sample' in col_lower:
            control_indicators.append(col)
    
    # Try to find controls based on these columns
    if control_indicators:
        for col in control_indicators:
            # Look for positive controls
            pos_mask = df[col].astype(str).str.lower().str.contains('positive|pos|pos_control|high')
            if pos_mask.any():
                controls["positive_controls"] = df.loc[pos_mask].index.tolist()
            
            # Look for negative controls
            neg_mask = df[col].astype(str).str.lower().str.contains('negative|neg|neg_control|low|blank')
            if neg_mask.any():
                controls["negative_controls"] = df.loc[neg_mask].index.tolist()
    
    # If we didn't find controls, look in any columns that might contain identifiers
    if not controls["positive_controls"] and not controls["negative_controls"]:
        for col in df.columns:
            # Skip numeric columns
            if df[col].dtype in [np.float64, np.int64]:
                continue
                
            # Look for positive controls in non-numeric columns
            pos_mask = df[col].astype(str).str.lower().str.contains('positive|pos|pos_control|high')
            if pos_mask.any():
                controls["positive_controls"] = df.loc[pos_mask].index.tolist()
            
            # Look for negative controls in non-numeric columns
            neg_mask = df[col].astype(str).str.lower().str.contains('negative|neg|neg_control|low|blank')
            if neg_mask.any():
                controls["negative_controls"] = df.loc[neg_mask].index.tolist()
    
    # If API key is available, use Claude for more sophisticated control identification
    if api_key and (not controls["positive_controls"] or not controls["negative_controls"]):
        try:
            # Create sample of data to send to Claude
            data_sample = df.to_string(max_rows=50)
            
            client = anthropic.Anthropic(api_key=api_key)
            prompt = f"""
            I have an experimental assay dataset that should include positive and negative controls.
            
            Here's the data:
            {data_sample}
            
            Please help me identify:
            1. Which rows are positive controls?
            2. Which rows are negative controls?
            
            Look for patterns like:
            - Rows with labels containing "positive", "pos", "negative", "neg"
            - Rows with consistently high or low values across measurements
            - Any standard naming patterns for controls
            
            Return your answer as JSON only:
            {{
                "positive_controls": [list of row indices or identifiers],
                "negative_controls": [list of row indices or identifiers],
                "reasoning": "brief explanation of how you identified controls"
            }}
            """
            
            message = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=1000,
                temperature=0,
                system="You are a data scientist specializing in experimental biology. Respond with JSON only.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            response = message.content[0].text
            
            # Try to extract JSON from the response
            try:
                json_match = re.search(r'\{.*\}', response.replace('\n', ''), re.DOTALL)
                if json_match:
                    ai_controls = json.loads(json_match.group(0))
                    controls.update(ai_controls)
                    # Remove 'reasoning' from controls dict
                    if 'reasoning' in controls:
                        reasoning = controls.pop('reasoning')
                        st.info(f"Control identification reasoning: {reasoning}")
            except Exception as json_err:
                st.warning(f"Could not parse AI control identification: {str(json_err)}")
                
        except Exception as e:
            st.warning(f"Could not use AI for control identification: {str(e)}")
    
    # Now validate controls and calculate statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Function to safely extract controls and calculate stats
    def calculate_control_stats(control_list, df, numeric_cols):
        stats = {}
        
        # Try to extract the control rows
        if not control_list:
            return stats
        
        # Extract controls either by index or by matching in the dataframe
        try:
            if isinstance(control_list[0], int):
                # These are indices
                control_df = df.iloc[control_list]
            else:
                # These might be identifiers
                # Try to match them against indices
                try:
                    control_df = df.loc[control_list]
                except:
                    # Try to match against ID-like columns
                    for col in df.columns:
                        if df[col].dtype not in [np.float64, np.int64]:  # Skip numeric columns
                            matched_rows = df[df[col].isin(control_list)]
                            if not matched_rows.empty:
                                control_df = matched_rows
                                break
                    else:
                        # No matches found
                        return stats
            
            # Calculate statistics for each numeric column
            for col in numeric_cols:
                try:
                    values = control_df[col].astype(float)
                    mean = values.mean()
                    std = values.std()
                    cv = (std / mean) * 100 if mean != 0 else float('inf')
                    
                    stats[col] = {
                        "mean": float(mean),
                        "std": float(std),
                        "cv": float(cv),
                        "n": len(values)
                    }
                except:
                    # Skip columns that can't be processed
                    pass
                    
        except Exception as e:
            st.warning(f"Error calculating control statistics: {str(e)}")
        
        return stats
    
    # Calculate statistics for positive and negative controls
    control_stats["positive"] = calculate_control_stats(controls["positive_controls"], df, numeric_cols)
    control_stats["negative"] = calculate_control_stats(controls["negative_controls"], df, numeric_cols)
    
    # Validate controls
    valid_controls = True
    
    # Check if we have both positive and negative controls
    if not control_stats["positive"] or not control_stats["negative"]:
        st.warning("Missing positive or negative controls")
        valid_controls = False
    
    # Check CV values for each measurement
    for col in numeric_cols:
        if col in control_stats["positive"] and col in control_stats["negative"]:
            pos_cv = control_stats["positive"][col].get("cv", float('inf'))
            neg_cv = control_stats["negative"][col].get("cv", float('inf'))
            
            # Flag high variation in controls
            if pos_cv > 20:
                st.warning(f"High variation in positive controls for {col} (CV: {pos_cv:.1f}%)")
                valid_controls = False
                
            if neg_cv > 20:
                st.warning(f"High variation in negative controls for {col} (CV: {neg_cv:.1f}%)")
                valid_controls = False
            
            # Check signal to background ratio
            pos_mean = control_stats["positive"][col].get("mean", 0)
            neg_mean = control_stats["negative"][col].get("mean", 0)
            
            if neg_mean != 0 and pos_mean / neg_mean < 2:
                st.warning(f"Low signal-to-background ratio for {col} ({pos_mean/neg_mean:.2f})")
                valid_controls = False
    
    return controls, control_stats

def normalize_data(df, method, controls, control_stats):
    """
    Apply the requested normalization method to the data.
    """
    if method == "None":
        return df.copy()
    
    result_df = df.copy()
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if method == "Z-score":
        st.write("Applying Z-score normalization...")
        for col in numeric_cols:
            # Check if column has sufficient variance
            std = df[col].std()
            if std == 0:
                st.warning(f"Column {col} has zero standard deviation - skipping normalization")
                continue
                
            # Apply Z-score normalization
            mean = df[col].mean()
            result_df[col] = (df[col] - mean) / std
    
    elif method == "Percent activity":
        st.write("Applying percent activity normalization...")
        
        # Check if we have valid control stats
        if not control_stats or not control_stats.get("positive") or not control_stats.get("negative"):
            st.warning("Invalid or missing control statistics. Using min/max for percent activity.")
            
            # Fall back to min/max normalization
            for col in numeric_cols:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val - min_val != 0:
                    result_df[col] = ((df[col] - min_val) / (max_val - min_val)) * 100
                else:
                    st.warning(f"Column {col} has zero range - skipping normalization")
        else:
            # Use control values for normalization
            for col in numeric_cols:
                if col in control_stats["positive"] and col in control_stats["negative"]:
                    pos_mean = control_stats["positive"][col].get("mean")
                    neg_mean = control_stats["negative"][col].get("mean")
                    
                    if pos_mean is not None and neg_mean is not None and pos_mean - neg_mean != 0:
                        result_df[col] = ((df[col] - neg_mean) / (pos_mean - neg_mean)) * 100
                    else:
                        st.warning(f"Invalid control values for {col} - skipping normalization")
                else:
                    st.warning(f"No control statistics for {col} - skipping normalization")
    
    return result_df

def unify_data_labels(dataframes):
    """
    Identify common data labels across all files.
    """
    # Extract all column names from all dataframes
    all_columns = []
    for df in dataframes:
        all_columns.extend(df.columns.tolist())
    
    # Count occurrences of each column
    column_counts = {}
    for col in all_columns:
        if col in column_counts:
            column_counts[col] += 1
        else:
            column_counts[col] = 1
    
    # Find columns that appear in multiple dataframes
    common_columns = [col for col, count in column_counts.items() if count > 1]
    
    return common_columns

def combine_samples_across_files(processed_files):
    """
    Check if samples appear in multiple files and join them if so.
    """
    if len(processed_files) <= 1:
        return None
    
    # Find candidate ID columns in each file
    id_columns = {}
    
    for idx, (filename, df, _) in enumerate(processed_files):
        # Try to find ID column using heuristics
        for col in df.columns:
            col_lower = str(col).lower()
            if 'id' in col_lower or 'sample' in col_lower:
                id_columns[idx] = col
                break
    
    # If we couldn't find ID columns for all files, return None
    if len(id_columns) != len(processed_files):
        st.warning("Could not identify sample ID columns in all files. Cannot join samples.")
        return None
    
    # Check for common samples
    sample_maps = []
    
    for idx, (filename, df, _) in enumerate(processed_files):
        if idx in id_columns:
            id_col = id_columns[idx]
            # Map sample IDs to row indices
            sample_maps.append({
                'file_idx': idx,
                'id_col': id_col,
                'samples': set(df[id_col].astype(str))
            })
    
    # Find samples that appear in multiple files
    all_samples = set()
    for sample_map in sample_maps:
        all_samples.update(sample_map['samples'])
    
    # Count occurrences of each sample
    sample_counts = {}
    for sample_map in sample_maps:
        for sample in sample_map['samples']:
            if sample in sample_counts:
                sample_counts[sample] += 1
            else:
                sample_counts[sample] = 1
    
    # Identify samples that appear in multiple files
    common_samples = {sample for sample, count in sample_counts.items() if count > 1}
    
    if not common_samples:
        st.info("No samples found in multiple files.")
        return None
    
    st.info(f"Found {len(common_samples)} samples that appear in multiple files.")
    
    # Join data for common samples
    combined_data = []
    
    for sample in common_samples:
        sample_data = {'Sample_ID': sample}
        
        for idx, (filename, df, _) in enumerate(processed_files):
            if idx in id_columns:
                id_col = id_columns[idx]
                
                # Find rows with this sample
                matching_rows = df[df[id_col].astype(str) == sample]
                
                if not matching_rows.empty:
                    # Get first matching row
                    row = matching_rows.iloc[0]
                    
                    # Add all columns except ID column
                    for col in df.columns:
                        if col != id_col:
                            # Use original column name if unique, otherwise add file identifier
                            col_name = col if col not in sample_data else f"{col}_{filename}"
                            sample_data[col_name] = row[col]
        
        combined_data.append(sample_data)
    
    # Create combined dataframe
    combined_df = pd.DataFrame(combined_data)
    
    return combined_df

if __name__ == "__main__":
    main()
