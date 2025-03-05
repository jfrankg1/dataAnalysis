import streamlit as st
import pandas as pd
import numpy as np
import anthropic
import os
import json
import re
from io import StringIO
import base64
import uuid

# Page configuration
st.set_page_config(page_title="Assay Data Normalizer", layout="wide")

def main():
    st.title("Experimental Assay Data Normalizer")
    st.write("Upload your CSV files, select normalization method, and get processed results with AI-assisted analysis.")
    
    # API key input (with warning about security)
    with st.expander("Claude API Configuration"):
        st.warning("Note: For production use, you should use environment variables or secure methods to store API keys.")
        api_key = st.text_input("Enter your Anthropic API Key", type="password")
    
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
        if not api_key:
            st.error("Please enter your Anthropic API key to use AI-assisted analysis.")
            return
            
        # Step 1: Process individual files
        processed_dfs = []
        intermediate_dfs = []
        
        for uploaded_file in uploaded_files:
            with st.expander(f"Processing {uploaded_file.name}"):
                # Process the individual file
                result = process_individual_file(uploaded_file, norm_method, api_key)
                
                if result["status"] == "success":
                    st.success(f"Successfully processed {uploaded_file.name}")
                    processed_dfs.append((uploaded_file.name, result["processed_df"]))
                    intermediate_dfs.append((uploaded_file.name, result["intermediate_df"], result["data_labels"], result["label_categories"]))
                else:
                    st.error(f"Failed to process {uploaded_file.name}: {result['error']}")
        
        # Step 2: Process files collectively if multiple files were uploaded and processed
        if len(processed_dfs) > 1:
            with st.expander("Cross-file Analysis"):
                collective_result = process_files_collectively(intermediate_dfs, norm_method, api_key)
                
                if collective_result["status"] == "success":
                    st.success("Successfully performed cross-file analysis and merging")
                    # Replace individual processed files with collectively processed ones if appropriate
                    if collective_result["merged_files"]:
                        processed_dfs = collective_result["output_dfs"]
                else:
                    st.warning(f"Cross-file analysis note: {collective_result['message']}")
        
        # Display and provide download links for processed files
        if processed_dfs:
            st.success("Processing complete! Results are available below.")
            display_results(processed_dfs)

def process_individual_file(file, norm_method, api_key):
    """Process a single data file according to the workflow"""
    try:
        st.write(f"### Analyzing {file.name}")
        
        # Read the CSV file
        df = pd.read_csv(file)
        original_df = df.copy()  # Keep a copy of the original data
        
        # Display original data
        st.write("Original Data:")
        st.dataframe(df.head())
        
        # Step 1: Identify data labels
        label_analysis = identify_data_labels(df, api_key)
        
        if label_analysis["status"] == "error":
            return {"status": "error", "error": label_analysis["error"]}
        
        # Step 2: Handle data orientation (transpose if needed)
        intermediate_df = handle_data_orientation(df, label_analysis, api_key)
        
        if intermediate_df is None:
            return {"status": "error", "error": "Could not determine data orientation"}
        
        st.write("Data with identified labels:")
        st.dataframe(intermediate_df.head())
        
        # Step 3: Categorize data labels
        label_categories = categorize_data_labels(intermediate_df, label_analysis["labels"], api_key)
        
        st.write("Label Categories:")
        categories_df = pd.DataFrame({
            'Label': list(label_categories.keys()),
            'Category': list(label_categories.values())
        })
        st.dataframe(categories_df)
        
        # Step 4: Analyze sample characteristics
        characteristics_analysis = analyze_sample_characteristics(intermediate_df, label_categories, api_key)
        
        st.write("Sample Characteristics Analysis:")
        st.write(characteristics_analysis["summary"])
        
        # Step 5: Identify and analyze controls
        controls_analysis = analyze_controls(intermediate_df, label_categories, api_key)
        
        st.write("Control Analysis:")
        st.write(controls_analysis["summary"])
        
        if not controls_analysis["controls_usable"]:
            st.warning("Controls may not be reliable for normalization. Proceeding with caution.")
        
        # Step 6: Normalize the data
        if norm_method != "None":
            normalized_df = normalize_data(
                intermediate_df, 
                norm_method, 
                label_categories,
                controls_analysis,
                api_key
            )
            
            st.write("Normalized Data:")
            st.dataframe(normalized_df.head())
        else:
            normalized_df = intermediate_df.copy()
            st.write("No normalization applied.")
        
        # Add metadata
        normalized_df.attrs["file_name"] = file.name
        normalized_df.attrs["normalization_method"] = norm_method
        normalized_df.attrs["label_categories"] = label_categories
        
        return {
            "status": "success",
            "processed_df": normalized_df,
            "intermediate_df": intermediate_df,
            "data_labels": label_analysis["labels"],
            "label_categories": label_categories
        }
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return {"status": "error", "error": str(e)}

def identify_data_labels(df, api_key):
    """Use Claude to identify data labels in the file"""
    st.write("Identifying data labels...")
    
    # Get the first few rows and columns for analysis
    sample_data = df.head(10).to_string()
    
    prompt = f"""
    I'm analyzing an experimental data file and need to identify whether the first non-empty row contains data labels, or if the first non-empty column contains data labels.

    Here's a sample of the data:
    {sample_data}

    Please analyze this data and:
    1. Determine if the first non-empty row likely contains data labels (column headers)
    2. If not, determine if the first non-empty column likely contains data labels (row indices)
    3. If you can identify data labels, list them
    4. If you cannot identify data labels in either place, explain why

    Return your analysis in this JSON format:
    ```
    {
      "has_row_labels": true/false,
      "has_column_labels": true/false,
      "labels": [list of identified labels],
      "confidence": "high/medium/low",
      "reasoning": "your explanation"
    }
    ```

    Please ensure your response is valid, parseable JSON.
    """
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1500,
            temperature=0,
            system="You are a helpful AI assistant specialized in analyzing experimental data structures. Always return valid, parseable JSON.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        response = message.content[0].text
        
        # Try to extract JSON from the response
        try:
            # Find JSON in the response (it might be wrapped in markdown code blocks)
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response
                
            analysis = json.loads(json_str)
            
            # Validate required keys
            required_keys = ["has_row_labels", "has_column_labels", "labels", "confidence", "reasoning"]
            for key in required_keys:
                if key not in analysis:
                    st.warning(f"Claude's response missing key: {key}. Using fallback.")
                    raise KeyError(f"Missing required key: {key}")
            
            st.write(f"Label identification complete (Confidence: {analysis['confidence']})")
            st.write(f"Found {len(analysis['labels'])} labels")
            
            return {
                "status": "success", 
                "has_row_labels": analysis["has_row_labels"],
                "has_column_labels": analysis["has_column_labels"],
                "labels": analysis["labels"],
                "reasoning": analysis["reasoning"]
            }
            
        except Exception as e:
            st.error(f"Error parsing Claude's response: {str(e)}")
            st.write("Raw response:", response)
            
            # Fallback: Assume column names are labels
            return {
                "status": "success",
                "has_row_labels": False,
                "has_column_labels": True,
                "labels": df.columns.tolist(),
                "reasoning": "Fallback: using column names as labels"
            }
            
    except Exception as e:
        st.error(f"Error connecting to Claude API: {str(e)}")
        return {"status": "error", "error": str(e)}

def handle_data_orientation(df, label_analysis, api_key):
    """Handle data orientation based on label analysis, transpose if needed"""
    if label_analysis["has_column_labels"]:
        # Data is already in the right orientation
        return df.copy()
    elif label_analysis["has_row_labels"]:
        # Need to transpose the data
        st.write("Transposing data to put labels in the header row...")
        
        # Find the first non-empty row that contains labels
        first_non_empty = None
        for i, row in df.iterrows():
            if not row.isna().all():
                first_non_empty = i
                break
        
        if first_non_empty is None:
            return None
        
        # Extract the labels from this row
        labels = df.iloc[first_non_empty].tolist()
        
        # Create a new dataframe with the data below this row
        data_rows = df.iloc[first_non_empty+1:].values
        
        # Create a transposed dataframe
        transposed_df = pd.DataFrame(data_rows, columns=labels)
        
        # Clean up column names
        transposed_df.columns = [str(col).strip() for col in transposed_df.columns]
        
        return transposed_df
    else:
        st.error("Could not identify data labels in either rows or columns.")
        return None

def categorize_data_labels(df, labels, api_key):
    """Use Claude to categorize data labels as sample characteristics or data outputs"""
    st.write("Categorizing data labels...")
    
    # Get a sample of the data for each label
    sample_data = {}
    for label in labels:
        if label in df.columns:
            sample_data[label] = df[label].head(5).tolist()
    
    # Prepare the sample data for Claude
    sample_json = json.dumps(sample_data, default=str)
    
    prompt = f"""
    I have data labels from an experimental assay dataset and need to categorize each as either:
    1. "Sample Characteristic": Labels that describe properties of samples (e.g., sample ID, treatment, concentration, time point)
    2. "Data Output": Labels that contain actual measurements or experimental results

    Here are the labels: {labels}

    Here's a sample of the data for each label:
    {sample_json}

    Please analyze each label and categorize it. Consider:
    - Labels containing words like "ID", "name", "treatment", "concentration", "time" likely refer to sample characteristics
    - Labels containing words like "activity", "value", "signal", "measurement", "response" likely refer to data outputs
    - Look at the actual values to help determine the category (e.g., numeric values are often data outputs, text/categorical values are often sample characteristics)

    Return your categorization as a JSON dictionary with label names as keys and categories as values:
    ```
    {
      "label1": "Sample Characteristic",
      "label2": "Data Output",
      ...
    }
    ```
    
    Please ensure your response is valid, parseable JSON.
    """
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1500,
            temperature=0,
            system="You are a helpful AI assistant specialized in experimental data analysis. Always return valid, parseable JSON.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        response = message.content[0].text
        
        # Try to extract JSON from the response
        try:
            # Find JSON in the response (it might be wrapped in markdown code blocks)
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response
                
            categories = json.loads(json_str)
            
            # Validate that all labels are categorized
            for label in labels:
                if label not in categories:
                    st.warning(f"Label '{label}' was not categorized by Claude. Assuming it's a data output.")
                    categories[label] = "Data Output"
            
            st.write("Label categorization complete")
            return categories
            
        except Exception as e:
            st.error(f"Error parsing Claude's categorization: {str(e)}")
            st.write("Raw response:", response)
            
            # Fallback: attempt to categorize based on data type
            categories = {}
            for label in labels:
                if label in df.columns:
                    if pd.api.types.is_numeric_dtype(df[label]):
                        categories[label] = "Data Output"
                    else:
                        categories[label] = "Sample Characteristic"
                else:
                    categories[label] = "Unknown"
            
            st.write("Using fallback categorization based on data types")
            return categories
            
    except Exception as e:
        st.error(f"Error connecting to Claude API: {str(e)}")
        
        # Fallback categorization
        categories = {}
        for label in labels:
            if label in df.columns:
                if pd.api.types.is_numeric_dtype(df[label]):
                    categories[label] = "Data Output"
                else:
                    categories[label] = "Sample Characteristic"
            else:
                categories[label] = "Unknown"
        
        st.write("Using fallback categorization based on data types")
        return categories

def analyze_sample_characteristics(df, label_categories, api_key):
    """Analyze sample characteristics to inform the analysis"""
    st.write("Analyzing sample characteristics...")
    
    # Extract sample characteristic columns
    characteristic_cols = [col for col, category in label_categories.items() 
                          if category == "Sample Characteristic" and col in df.columns]
    
    if not characteristic_cols:
        return {
            "summary": "No sample characteristics identified for analysis.",
            "insights": []
        }
    
    # Prepare data for Claude
    characteristics_data = df[characteristic_cols].head(20).to_string()
    
    prompt = f"""
    I have sample characteristic data from an experimental assay. Please analyze this data to extract insights that could inform the analysis.

    The following columns are sample characteristics:
    {characteristic_cols}

    Here's a sample of the data:
    {characteristics_data}

    Please analyze this data and:
    1. Identify any patterns or groupings in the samples
    2. Detect any potential experimental design features (e.g., dose responses, time series, replicates)
    3. Flag any issues or anomalies in the sample characteristics
    4. Suggest how this information could inform data normalization

    Return your analysis in this JSON format:
    ```
    {{
      "summary": "Brief summary of findings",
      "insights": [
        {{
          "type": "pattern/design/issue/suggestion",
          "description": "Description of the insight",
          "relevance": "How this affects normalization or analysis"
        }},
        ...
      ]
    }}
    ```

    Please ensure your response is valid, parseable JSON.
    """
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=2000,
            temperature=0,
            system="You are a helpful AI assistant specialized in experimental design and data analysis. Always return valid, parseable JSON.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        response = message.content[0].text
        
        # Try to extract JSON from the response
        try:
            # Find JSON in the response
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response
                
            analysis = json.loads(json_str)
            
            # Validate required keys
            if "summary" not in analysis or "insights" not in analysis:
                raise KeyError("Missing required keys in analysis")
            
            return analysis
            
        except Exception as e:
            st.error(f"Error parsing Claude's sample analysis: {str(e)}")
            
            # Return a simplified result
            return {
                "summary": "Analysis completed with limited insights due to parsing issues.",
                "insights": []
            }
            
    except Exception as e:
        st.error(f"Error connecting to Claude API: {str(e)}")
        
        # Return a simplified result
        return {
            "summary": "Could not analyze sample characteristics due to API error.",
            "insights": []
        }

def analyze_controls(df, label_categories, api_key):
    """Identify and analyze controls in the dataset"""
    st.write("Identifying and analyzing controls...")
    
    # Extract data output columns
    data_cols = [col for col, category in label_categories.items() 
                if category == "Data Output" and col in df.columns]
    
    if not data_cols:
        return {
            "summary": "No data output columns identified for control analysis.",
            "positive_controls": None,
            "negative_controls": None,
            "statistics": {},
            "controls_usable": False
        }
    
    # Extract sample characteristic columns (might help identify controls)
    characteristic_cols = [col for col, category in label_categories.items() 
                          if category == "Sample Characteristic" and col in df.columns]
    
    # Calculate basic statistics for data columns
    stats = {}
    for col in data_cols:
        numeric_data = pd.to_numeric(df[col], errors='coerce')
        stats[col] = {
            "mean": numeric_data.mean(),
            "median": numeric_data.median(),
            "std": numeric_data.std(),
            "cv": (numeric_data.std() / numeric_data.mean()) * 100 if numeric_data.mean() != 0 else None,
            "min": numeric_data.min(),
            "max": numeric_data.max()
        }
    
    # Prepare data for Claude
    data_sample = df.head(20).to_string()
    stats_json = json.dumps({k: {k2: v2 for k2, v2 in v.items() if v2 is not None} for k, v in stats.items()}, default=str)
    
    prompt = f"""
    I have experimental assay data and need to identify positive and negative controls, then analyze if they're suitable for normalization.

    Sample characteristic columns: {characteristic_cols}
    Data output columns: {data_cols}

    Here's a sample of the data:
    {data_sample}

    Statistics for data columns:
    {stats_json}

    Please:
    1. Identify which rows/values are likely positive controls (high values/activity)
    2. Identify which rows/values are likely negative controls (low values/background)
    3. Analyze the statistics of these controls to determine if they're reliable for normalization
    4. Consider acceptable ranges for coefficient of variation (CV) for controls (typically <20% is good)

    Look for:
    - Keywords like "pos", "pos_ctrl", "negative", "neg_ctrl", "control+", "control-" in sample characteristics
    - Min/max values that might represent controls
    - Patterns in well positions (A1, H12 often used for controls in plate formats)
    - Clustering of values that might indicate control groups

    Return your analysis in this JSON format:
    ```
    {{
      "summary": "Brief summary of control identification and quality",
      "positive_controls": {{
        "identification_method": "How positive controls were identified",
        "values": [list of identifiers or indices],
        "statistics": {{
          "mean": value,
          "std": value,
          "cv": value
        }}
      }},
      "negative_controls": {{
        "identification_method": "How negative controls were identified",
        "values": [list of identifiers or indices],
        "statistics": {{
          "mean": value,
          "std": value,
          "cv": value
        }}
      }},
      "controls_usable": true/false,
      "reasoning": "Explanation of control quality assessment"
    }}
    ```

    Please ensure your response is valid, parseable JSON.
    """
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=2000,
            temperature=0,
            system="You are a helpful AI assistant specialized in experimental design and control analysis. Always return valid, parseable JSON.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        response = message.content[0].text
        
        # Try to extract JSON from the response
        try:
            # Find JSON in the response
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response
                
            analysis = json.loads(json_str)
            
            # Validate required keys
            required_keys = ["summary", "positive_controls", "negative_controls", "controls_usable"]
            for key in required_keys:
                if key not in analysis:
                    st.warning(f"Claude's control analysis missing key: {key}. Using fallback.")
                    raise KeyError(f"Missing required key: {key}")
            
            return analysis
            
        except Exception as e:
            st.error(f"Error parsing Claude's control analysis: {str(e)}")
            
            # Fallback analysis
            fallback_positive = None
            fallback_negative = None
            
            # Simple fallback: use max/min values as proxies for controls
            for col in data_cols:
                numeric_data = pd.to_numeric(df[col], errors='coerce')
                if fallback_positive is None:
                    fallback_positive = {
                        "identification_method": "Fallback: using maximum values as proxy for positive controls",
                        "values": numeric_data.nlargest(3).index.tolist(),
                        "statistics": {
                            "mean": numeric_data.nlargest(3).mean(),
                            "std": numeric_data.nlargest(3).std(),
                            "cv": (numeric_data.nlargest(3).std() / numeric_data.nlargest(3).mean()) * 100 if numeric_data.nlargest(3).mean() != 0 else None
                        }
                    }
                
                if fallback_negative is None:
                    fallback_negative = {
                        "identification_method": "Fallback: using minimum values as proxy for negative controls",
                        "values": numeric_data.nsmallest(3).index.tolist(),
                        "statistics": {
                            "mean": numeric_data.nsmallest(3).mean(),
                            "std": numeric_data.nsmallest(3).std(),
                            "cv": (numeric_data.nsmallest(3).std() / numeric_data.nsmallest(3).mean()) * 100 if numeric_data.nsmallest(3).mean() != 0 else None
                        }
                    }
            
            return {
                "summary": "Fallback control analysis used. Controls identified based on value extremes.",
                "positive_controls": fallback_positive,
                "negative_controls": fallback_negative,
                "controls_usable": True,  # Assuming usable with caution
                "reasoning": "Fallback analysis: controls identified using maximum and minimum values as proxies."
            }
            
    except Exception as e:
        st.error(f"Error connecting to Claude API: {str(e)}")
        
        # Fallback analysis
        fallback_positive = None
        fallback_negative = None
        
        # Simple fallback
        for col in data_cols:
            numeric_data = pd.to_numeric(df[col], errors='coerce')
            if fallback_positive is None:
                fallback_positive = {
                    "identification_method": "Fallback: using maximum values as proxy for positive controls",
                    "values": numeric_data.nlargest(3).index.tolist(),
                    "statistics": {
                        "mean": numeric_data.nlargest(3).mean(),
                        "std": numeric_data.nlargest(3).std(),
                        "cv": (numeric_data.nlargest(3).std() / numeric_data.nlargest(3).mean()) * 100 if numeric_data.nlargest(3).mean() != 0 else None
                    }
                }
            
            if fallback_negative is None:
                fallback_negative = {
                    "identification_method": "Fallback: using minimum values as proxy for negative controls",
                    "values": numeric_data.nsmallest(3).index.tolist(),
                    "statistics": {
                        "mean": numeric_data.nsmallest(3).mean(),
                        "std": numeric_data.nsmallest(3).std(),
                        "cv": (numeric_data.nsmallest(3).std() / numeric_data.nsmallest(3).mean()) * 100 if numeric_data.nsmallest(3).mean() != 0 else None
                    }
                }
        
        return {
            "summary": "Fallback control analysis used. Controls identified based on value extremes.",
            "positive_controls": fallback_positive,
            "negative_controls": fallback_negative,
            "controls_usable": True,  # Assuming usable with caution
            "reasoning": "Fallback analysis: controls identified using maximum and minimum values as proxies."
        }

def normalize_data(df, method, label_categories, controls_analysis, api_key):
    """Apply the selected normalization method to the data"""
    st.write(f"Applying {method} normalization...")
    
    result_df = df.copy()
    
    # Get data output columns
    data_cols = [col for col, category in label_categories.items() 
                if category == "Data Output" and col in df.columns]
    
    if method == "Z-score":
        for col in data_cols:
            numeric_data = pd.to_numeric(df[col], errors='coerce')
            mean = numeric_data.mean()
            std = numeric_data.std()
            if std != 0:  # Avoid division by zero
                result_df[col] = (numeric_data - mean) / std
                st.write(f"✓ Z-score normalized: {col}")
            else:
                st.warning(f"Column {col} has zero standard deviation, skipping normalization.")
    
    elif method == "Percent activity":
        # Try to use the identified controls for normalization
        try:
            pos_controls = controls_analysis.get("positive_controls", {})
            neg_controls = controls_analysis.get("negative_controls", {})
            
            for col in data_cols:
                numeric_data = pd.to_numeric(df[col], errors='coerce')
                
                # Get positive control mean
                if pos_controls and pos_controls.get("statistics", {}).get("mean") is not None:
                    pos_mean = pos_controls["statistics"]["mean"]
                else:
                    pos_mean = numeric_data.max()
                    st.warning(f"Using maximum value as positive control for {col}")
                
                # Get negative control mean
                if neg_controls and neg_controls.get("statistics", {}).get("mean") is not None:
                    neg_mean = neg_controls["statistics"]["mean"]
                else:
                    neg_mean = numeric_data.min()
                    st.warning(f"Using minimum value as negative control for {col}")
                
                # Apply percent activity normalization
                if pos_mean - neg_mean != 0:  # Avoid division by zero
                    result_df[col] = ((numeric_data - neg_mean) / (pos_mean - neg_mean)) * 100
                    st.write(f"✓ Percent activity normalized: {col}")
                else:
                    st.warning(f"Column {col} has identical max/min values, skipping normalization.")
                
        except Exception as e:
            st.error(f"Error applying percent activity normalization: {str(e)}")
            
            # Fallback to simple min/max normalization
            for col in data_cols:
                numeric_data = pd.to_numeric(df[col], errors='coerce')
                min_val = numeric_data.min()
                max_val = numeric_data.max()
                if max_val - min_val != 0:  # Avoid division by zero
                    result_df[col] = ((numeric_data - min_val) / (max_val - min_val)) * 100
                    st.write(f"✓ Fallback percent activity normalized: {col}")
                else:
                    st.warning(f"Column {col} has identical min/max values, skipping normalization.")
    
    # Add normalization metadata
    result_df.attrs["normalization_method"] = method
    result_df.attrs["normalized_columns"] = data_cols
    
    return result_df

def process_files_collectively(intermediate_dfs, norm_method, api_key):
    """Process multiple files collectively to unify labels and merge related data"""
    if not intermediate_dfs or len(intermediate_dfs) < 2:
        return {
            "status": "success",
            "message": "Only one file processed, no collective analysis needed.",
            "merged_files": False,
            "output_dfs": []
        }
    
    st.write("### Analyzing files collectively")
    
    # Extract file info
    file_info = []
    for file_name, df, labels, categories in intermediate_dfs:
        # Get sample characteristic columns
        char_cols = [col for col, category in categories.items() 
                    if category == "Sample Characteristic" and col in df.columns]
        
        # Get data output columns
        data_cols = [col for col, category in categories.items() 
                    if category == "Data Output" and col in df.columns]
        
        file_info.append({
            "file_name": file_name,
            "df": df,
            "labels": labels,
            "categories": categories,
            "sample_chars": char_cols,
            "data_cols": data_cols,
            "sample_count": len(df)
        })
    
    # Analyze label similarities across files
    label_analysis = analyze_labels_across_files(file_info, api_key)
    
    if not label_analysis["unified_labels_possible"]:
        return {
            "status": "success",
            "message": "Files have significantly different structures. Processing individually.",
            "merged_files": False,
            "output_dfs": [(info["file_name"], info["df"]) for info in file_info]
        }
    
    # Check if files contain data for the same samples
    sample_analysis = analyze_samples_across_files(file_info, label_analysis, api_key)
    
    if not sample_analysis["shared_samples"]:
        st.write("No shared samples found across files. Processing files individually with unified labels.")
        
        # Use unified labels but don't merge files
        output_dfs = []
        for info in file_info:
            # Rename columns according to mapping if needed
            renamed_df = info["df"].rename(columns=label_analysis["label_mapping"].get(info["file_name"], {}))
            output_dfs.append((info["file_name"], renamed_df))
        
        return {
            "status": "success",
            "message": "Unified labels applied across files but no shared samples for merging.",
            "merged_files": False,
            "output_dfs": output_dfs
        }
    
    # Merge files with shared samples
    st.write(f"Found {len(sample_analysis['sample_groups'])} groups of related samples across files. Merging data...")
    
    merged_dfs = merge_related_files(file_info, sample_analysis, label_analysis)
    
    return {
        "status": "success",
        "message": f"Successfully merged related data across {len(file_info)} files.",
        "merged_files": True,
        "output_dfs": merged_dfs
    }

def analyze_labels_across_files(file_info, api_key):
    """Analyze label similarities across files and create a unified set"""
    st.write("Analyzing label similarities across files...")
    
    # Collect all labels from all files
    all_labels = {}
    for info in file_info:
        all_labels[info["file_name"]] = {
            "sample_chars": info["sample_chars"],
            "data_cols": info["data_cols"]
        }
    
    # Prepare the prompt for Claude
    prompt = f"""
    I have multiple experimental data files that may be related. I need to analyze their column labels to:
    1. Determine if they share similar data structures
    2. Create a unified set of labels for consistent analysis
    3. Map the original labels to the unified set

    Here are the labels from each file:
    {json.dumps(all_labels, indent=2)}

    Please analyze these labels and:
    1. Identify semantically similar labels across files (e.g., "sample_id" and "sampleID" likely refer to the same thing)
    2. Create a unified naming scheme for all labels
    3. Provide a mapping from original labels to unified labels for each file
    4. Determine if the files can reasonably be processed with a unified label set

    Return your analysis in this JSON format:
    ```
    {{
      "unified_labels_possible": true/false,
      "reasoning": "Explanation of similarity assessment",
      "unified_labels": {{
        "sample_characteristics": ["list", "of", "unified", "characteristic", "labels"],
        "data_outputs": ["list", "of", "unified", "data", "labels"]
      }},
      "label_mapping": {{
        "file1.csv": {{"original_label1": "unified_label1", "original_label2": "unified_label2", ...}},
        "file2.csv": {{"original_label1": "unified_label1", "original_label3": "unified_label3", ...}},
        ...
      }}
    }}
    ```

    Please ensure your response is valid, parseable JSON.
    """
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=2500,
            temperature=0,
            system="You are a helpful AI assistant specialized in data harmonization and metadata analysis. Always return valid, parseable JSON.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        response = message.content[0].text
        
        # Try to extract JSON from the response
        try:
            # Find JSON in the response
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response
                
            analysis = json.loads(json_str)
            
            # Validate required keys
            required_keys = ["unified_labels_possible", "reasoning", "unified_labels", "label_mapping"]
            for key in required_keys:
                if key not in analysis:
                    st.warning(f"Claude's label analysis missing key: {key}. Using fallback.")
                    raise KeyError(f"Missing required key: {key}")
            
            st.write(f"Label unification possible: {analysis['unified_labels_possible']}")
            if analysis['unified_labels_possible']:
                st.write(f"Identified {len(analysis['unified_labels']['sample_characteristics'])} shared sample characteristics and {len(analysis['unified_labels']['data_outputs'])} shared data outputs.")
            
            return analysis
            
        except Exception as e:
            st.error(f"Error parsing Claude's label analysis: {str(e)}")
            
            # Fallback analysis: create a very basic unification
            unified_labels = {
                "sample_characteristics": [],
                "data_outputs": []
            }
            
            label_mapping = {}
            
            # Collect all unique labels
            all_char_labels = set()
            all_data_labels = set()
            
            for info in file_info:
                all_char_labels.update(info["sample_chars"])
                all_data_labels.update(info["data_cols"])
                
                # Initialize empty mapping for each file
                label_mapping[info["file_name"]] = {}
            
            # Use original labels as unified labels (no renaming)
            unified_labels["sample_characteristics"] = list(all_char_labels)
            unified_labels["data_outputs"] = list(all_data_labels)
            
            return {
                "unified_labels_possible": True,
                "reasoning": "Fallback analysis: using original labels as unified set.",
                "unified_labels": unified_labels,
                "label_mapping": label_mapping  # Empty mappings (no renaming)
            }
            
    except Exception as e:
        st.error(f"Error connecting to Claude API: {str(e)}")
        
        # Basic fallback analysis
        unified_labels = {
            "sample_characteristics": [],
            "data_outputs": []
        }
        
        label_mapping = {}
        
        # Collect all unique labels
        all_char_labels = set()
        all_data_labels = set()
        
        for info in file_info:
            all_char_labels.update(info["sample_chars"])
            all_data_labels.update(info["data_cols"])
            
            # Initialize empty mapping for each file
            label_mapping[info["file_name"]] = {}
        
        # Use original labels as unified labels (no renaming)
        unified_labels["sample_characteristics"] = list(all_char_labels)
        unified_labels["data_outputs"] = list(all_data_labels)
        
        return {
            "unified_labels_possible": True,
            "reasoning": "Fallback analysis: using original labels as unified set.",
            "unified_labels": unified_labels,
            "label_mapping": label_mapping  # Empty mappings (no renaming)
        }

def analyze_samples_across_files(file_info, label_analysis, api_key):
    """Analyze whether multiple files contain data for the same samples"""
    st.write("Analyzing if files contain data for the same samples...")
    
    # Extract sample characteristics from each file
    sample_data = {}
    for info in file_info:
        # Get a subset of sample characteristics for identification
        sample_chars = info["sample_chars"]
        if sample_chars:
            sample_data[info["file_name"]] = info["df"][sample_chars].head(20).to_dict(orient='records')
    
    # Prepare the prompt for Claude
    prompt = f"""
    I have multiple experimental data files and need to determine if they contain data for the same samples.

    Here's sample data from each file, showing their sample characteristics:
    {json.dumps(sample_data, indent=2)}

    From the label analysis, we have these unified sample characteristics:
    {json.dumps(label_analysis["unified_labels"]["sample_characteristics"], indent=2)}

    Please analyze if these files contain information about the same samples by:
    1. Looking for matching identifiers or combinations of characteristics
    2. Identifying potential connections between samples across files
    3. Grouping related samples that should be merged

    Return your analysis in this JSON format:
    ```
    {{
      "shared_samples": true/false,
      "reasoning": "Explanation of your analysis",
      "sample_id_columns": ["column1", "column2"],
      "sample_groups": [
        {{
          "group_id": 1,
          "samples": [
            {{"file": "file1.csv", "identifiers": {{"column1": "value1", "column2": "value2"}}}},
            {{"file": "file2.csv", "identifiers": {{"column1": "value1", "column2": "value2"}}}}
          ]
        }},
        ...
      ]
    }}
    ```

    The "sample_id_columns" should list columns that uniquely identify samples.
    The "sample_groups" should list groups of related samples across different files.
    
    Please ensure your response is valid, parseable JSON.
    """
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=2500,
            temperature=0,
            system="You are a helpful AI assistant specialized in data analysis and integration. Always return valid, parseable JSON.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        response = message.content[0].text
        
        # Try to extract JSON from the response
        try:
            # Find JSON in the response
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response
                
            analysis = json.loads(json_str)
            
            # Validate required keys
            required_keys = ["shared_samples", "reasoning", "sample_id_columns", "sample_groups"]
            for key in required_keys:
                if key not in analysis:
                    st.warning(f"Claude's sample analysis missing key: {key}. Using fallback.")
                    raise KeyError(f"Missing required key: {key}")
            
            st.write(f"Shared samples detected: {analysis['shared_samples']}")
            if analysis['shared_samples']:
                st.write(f"Identified {len(analysis['sample_groups'])} groups of related samples.")
                st.write(f"Sample identification based on: {', '.join(analysis['sample_id_columns'])}")
            
            return analysis
            
        except Exception as e:
            st.error(f"Error parsing Claude's sample analysis: {str(e)}")
            
            # Fallback: assume no shared samples
            return {
                "shared_samples": False,
                "reasoning": "Fallback analysis: could not reliably identify shared samples.",
                "sample_id_columns": [],
                "sample_groups": []
            }
            
    except Exception as e:
        st.error(f"Error connecting to Claude API: {str(e)}")
        
        # Fallback: assume no shared samples
        return {
            "shared_samples": False,
            "reasoning": "Fallback analysis: could not reliably identify shared samples.",
            "sample_id_columns": [],
            "sample_groups": []
        }

def merge_related_files(file_info, sample_analysis, label_analysis):
    """Merge data from related files based on shared samples"""
    merged_dfs = []
    
    # Create a dictionary of file DataFrames for easier access
    file_dfs = {info["file_name"]: info["df"] for info in file_info}
    
    # Process each group of related samples
    for group in sample_analysis["sample_groups"]:
        group_id = group["group_id"]
        samples = group["samples"]
        
        if len(samples) < 2:
            continue  # Skip groups with only one sample
        
        # Create a merged DataFrame for this group
        merged_data = {}
        first_file = True
        
        for sample in samples:
            file_name = sample["file"]
            identifiers = sample["identifiers"]
            
            if file_name not in file_dfs:
                continue
                
            df = file_dfs[file_name]
            
            # Find matching rows in the DataFrame
            # This is a simplified approach - in a real app, you would need more robust matching
            matches = df
            for col, value in identifiers.items():
                if col in df.columns:
                    matches = matches[matches[col] == value]
            
            if len(matches) == 0:
                continue
                
            # First file establishes the base merged data
            if first_file:
                for col in df.columns:
                    merged_data[col] = matches[col].values[0] if len(matches) > 0 else None
                first_file = False
            else:
                # Add data from additional files
                for col in df.columns:
                    # Don't overwrite sample ID columns
                    if col in sample_analysis["sample_id_columns"]:
                        continue
                    
                    # Add new data
                    if col not in merged_data and len(matches) > 0:
                        merged_data[col] = matches[col].values[0]
        
        # Create a DataFrame from the merged data
        if merged_data:
            merged_df = pd.DataFrame([merged_data])
            merged_df.attrs["source_files"] = [sample["file"] for sample in samples]
            merged_df.attrs["group_id"] = group_id
            
            merged_filename = f"merged_group_{group_id}.csv"
            merged_dfs.append((merged_filename, merged_df))
    
    # If no merges were performed, return the original DataFrames
    if not merged_dfs:
        st.warning("Could not merge files despite detecting shared samples. Using original files.")
        return [(info["file_name"], info["df"]) for info in file_info]
    
    return merged_dfs

def display_results(processed_dfs):
    """Display results and provide download links"""
    for filename, df in processed_dfs:
        st.subheader(f"Processed Data: {filename}")
        
        # Display basic info
        st.write(f"Rows: {len(df)}, Columns: {len(df.columns)}")
        
        # Show source files if this is a merged file
        if hasattr(df, 'attrs') and 'source_files' in df.attrs:
            st.write(f"Merged from: {', '.join(df.attrs['source_files'])}")
        
        # Show normalization method if available
        if hasattr(df, 'attrs') and 'normalization_method' in df.attrs:
            st.write(f"Normalization method: {df.attrs['normalization_method']}")
        
        # Display the processed data
        st.dataframe(df.head(10))
        
        # Create download link
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        processed_filename = f"processed_{filename}"
        href = f'<a href="data:file/csv;base64,{b64}" download="{processed_filename}">Download processed file</a>'
        st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
