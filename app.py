import streamlit as st
import pandas as pd
import io
import json
import re
from anthropic import Anthropic

def main():
    st.set_page_config(page_title="Experimental Data Unifier", layout="wide")
    
    st.title("Experimental Data and Protocol Unifier")
    st.markdown("""
    This app unifies experimental data with protocol information:
    1. Upload protocol files (text format) and data files (CSV, Excel)
    2. The app will use Claude API to extract protocol information
    3. Raw data will be extracted from experimental data files
    4. A unified CSV containing all relevant information will be created
    5. Preview the first 5 lines of the unified data
    6. Download the complete unified CSV file
    """)
    
    # File uploaders in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Protocol Files")
        protocol_files = st.file_uploader("Upload protocol files (text format)", 
                                         type=["txt"], 
                                         accept_multiple_files=True)
        st.caption("Protocol files should describe experimental methods, samples, and conditions")
    
    with col2:
        st.subheader("Data Files")
        data_files = st.file_uploader("Upload experimental data files", 
                                      type=["csv", "xlsx", "xls"], 
                                      accept_multiple_files=True)
        st.caption("Data files should contain experimental results with sample IDs")
    
    # API key input in an expander
    with st.expander("API Settings"):
        st.info("An Anthropic API key is required to extract protocol information using Claude")
        api_key = st.text_input("Enter your Anthropic API Key", type="password")
        model = st.selectbox("Select Claude Model", 
                            ["claude-3-7-sonnet-20250219", 
                             "claude-3-opus-20240229",
                             "claude-3-haiku-20240307"])
    
    # Process button
    if st.button("Process Files", type="primary") and (protocol_files or data_files):
        if not api_key and protocol_files:
            st.error("API key is required to process protocol files")
            return
        
        with st.spinner("Processing files..."):
            progress = st.progress(0)
            
            # Process protocol files
            if protocol_files:
                st.info("Extracting protocol information using Claude API...")
                protocol_info = extract_protocol_info(protocol_files, api_key, model)
                progress.progress(50)
            else:
                protocol_info = pd.DataFrame()
                progress.progress(50)
            
            # Process data files
            if data_files:
                st.info("Extracting experimental data...")
                experimental_data = extract_experimental_data(data_files)
                progress.progress(75)
            else:
                experimental_data = pd.DataFrame()
                progress.progress(75)
            
            # Merge data
            st.info("Unifying data...")
            unified_data = merge_data(protocol_info, experimental_data)
            progress.progress(100)
            
            # Display results
            if not unified_data.empty:
                st.success("Processing complete!")
                
                # Display preview
                st.subheader("Preview of Unified Data (First 5 rows)")
                st.dataframe(unified_data.head())
                
                # Data summary
                st.subheader("Data Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Rows", len(unified_data))
                with col2:
                    st.metric("Total Columns", len(unified_data.columns))
                
                # Download button
                csv = unified_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Complete Unified CSV",
                    data=csv,
                    file_name="unified_experimental_data.csv",
                    mime="text/csv",
                    key="download-csv"
                )
            else:
                st.error("No data was extracted. Please check your files and try again.")

def extract_protocol_info(protocol_files, api_key, model):
    """
    Extract protocol information from text files using Claude API
    """
    all_protocol_data = []
    
    # Initialize Anthropic client
    client = Anthropic(api_key=api_key)
    
    for file in protocol_files:
        with st.status(f"Processing {file.name}..."):
            # Read file content
            try:
                content = file.read().decode('utf-8')
            except UnicodeDecodeError:
                try:
                    content = file.read().decode('latin-1')
                except:
                    st.error(f"Could not decode {file.name}")
                    continue
            
            # Call Claude API
            try:
                st.write("Calling Claude API...")
                response = client.messages.create(
                    model=model,
                    max_tokens=4000,
                    messages=[
                        {
                            "role": "user", 
                            "content": f"""
                            Extract all experimental protocol information from this document.
                            
                            Please identify:
                            1. Sample IDs or names
                            2. Protocol name/title
                            3. Protocol steps in sequence
                            4. Reagents used with concentrations
                            5. Equipment used
                            6. Experimental conditions (temperature, time, etc.)
                            
                            Format your response as a clean JSON object:
                            {{
                                "sample_id": [list of sample IDs],
                                "protocol_name": "name of protocol",
                                "protocol_steps": [list of protocol steps],
                                "reagents": [list of reagents used],
                                "equipment": [list of equipment used],
                                "conditions": [list of experimental conditions]
                            }}
                            
                            If information is not available, use empty lists or empty strings.
                            
                            Protocol document content:
                            
                            {content}
                            """
                        }
                    ]
                )
                
                # Extract JSON from Claude's response
                assistant_message = response.content[0].text
                json_str = extract_json_from_text(assistant_message)
                
                try:
                    protocol_json = json.loads(json_str)
                    protocol_df = json_to_dataframe(protocol_json, file.name)
                    all_protocol_data.append(protocol_df)
                    st.write("✅ Successfully extracted protocol information")
                except json.JSONDecodeError:
                    st.error(f"Could not parse JSON from Claude's response")
                    st.code(json_str[:500] + "...", language="json")
            
            except Exception as e:
                st.error(f"Error calling Claude API: {str(e)}")
    
    # Combine all protocol dataframes
    if all_protocol_data:
        return pd.concat(all_protocol_data, ignore_index=True)
    
    return pd.DataFrame()

def extract_json_from_text(text):
    """
    Extract JSON from Claude's response text
    """
    # Try to extract JSON from code blocks
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.DOTALL)
    
    if json_match:
        return json_match.group(1).strip()
    
    # Try to find JSON object pattern
    json_match = re.search(r'({[\s\S]*})', text, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()
    
    # Return the entire text if no match
    return text

def json_to_dataframe(protocol_json, source_filename):
    """
    Convert protocol JSON to a pandas DataFrame
    """
    flattened_data = []
    
    # Get sample IDs or use a placeholder
    sample_ids = protocol_json.get('sample_id', [])
    if not sample_ids:
        sample_ids = ['unknown']
    
    for sample_id in sample_ids:
        row = {
            'sample_id': sample_id,
            'protocol_name': protocol_json.get('protocol_name', ''),
            'protocol_steps': '; '.join(protocol_json.get('protocol_steps', [])),
            'reagents': '; '.join(protocol_json.get('reagents', [])),
            'equipment': '; '.join(protocol_json.get('equipment', [])),
            'conditions': '; '.join(protocol_json.get('conditions', [])),
            'protocol_source': source_filename
        }
        flattened_data.append(row)
    
    return pd.DataFrame(flattened_data)

def extract_experimental_data(data_files):
    """
    Extract experimental data from CSV and Excel files
    """
    all_data = []
    
    for file in data_files:
        with st.status(f"Processing {file.name}..."):
            try:
                # Process based on file type
                if file.name.endswith('.csv'):
                    df = pd.read_csv(file)
                elif file.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file)
                else:
                    st.warning(f"Unsupported file type: {file.name}")
                    continue
                
                # Add source information
                df['data_source'] = file.name
                
                # Standardize column names
                df.columns = [standardize_column_name(col) for col in df.columns]
                
                # Try to identify sample IDs if not already present
                if 'sample_id' not in df.columns:
                    potential_id_cols = ['id', 'sample', 'name', 'sample_name', 'sampleid', 'sample_id']
                    for col in df.columns:
                        if col.lower() in potential_id_cols:
                            df = df.rename(columns={col: 'sample_id'})
                            st.info(f"Renamed column '{col}' to 'sample_id'")
                            break
                
                all_data.append(df)
                st.write(f"✅ Successfully processed data with {len(df)} rows and {len(df.columns)} columns")
                
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
    
    # Combine all data
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    
    return pd.DataFrame()

def standardize_column_name(col_name):
    """
    Standardize column names to snake_case
    """
    # Convert to string in case of numeric column names
    col_name = str(col_name)
    
    # Replace spaces and special characters with underscores
    standardized = re.sub(r'[^a-zA-Z0-9]', '_', col_name)
    
    # Convert to lowercase
    standardized = standardized.lower()
    
    # Replace multiple underscores with a single one
    standardized = re.sub(r'_+', '_', standardized)
    
    # Remove leading and trailing underscores
    standardized = standardized.strip('_')
    
    return standardized

def merge_data(protocol_df, experimental_df):
    """
    Merge protocol and experimental data into a unified dataframe
    """
    # If one is empty, return the other
    if protocol_df.empty and experimental_df.empty:
        return pd.DataFrame()
    elif protocol_df.empty:
        return experimental_df
    elif experimental_df.empty:
        return protocol_df
    
    # Try to merge on 'sample_id' if it exists in both
    if 'sample_id' in protocol_df.columns and 'sample_id' in experimental_df.columns:
        st.info("Merging data based on sample IDs")
        merged_df = pd.merge(experimental_df, protocol_df, on='sample_id', how='outer')
    else:
        # If no common key, create a cross join with prefixes
        st.warning("No common 'sample_id' found for merging. Creating a combined dataset without direct sample matching.")
        
        # Add prefixes to avoid column name collisions
        protocol_df = protocol_df.add_prefix('protocol_')
        if 'protocol_sample_id' in protocol_df.columns:
            protocol_df = protocol_df.rename(columns={'protocol_sample_id': 'sample_id'})
            
        experimental_df = experimental_df.add_prefix('data_')
        if 'data_sample_id' in experimental_df.columns:
            experimental_df = experimental_df.rename(columns={'data_sample_id': 'sample_id'})
        
        # Create a cross join
        merged_df = create_cross_join(experimental_df, protocol_df)
    
    return merged_df

def create_cross_join(df1, df2):
    """
    Create a cross join (Cartesian product) of two dataframes
    """
    df1['_key'] = 1
    df2['_key'] = 1
    
    result = pd.merge(df1, df2, on='_key').drop('_key', axis=1)
    
    return result

if __name__ == "__main__":
    main()
