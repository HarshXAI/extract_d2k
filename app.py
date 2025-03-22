import streamlit as st
import json
import pandas as pd
from enhanced_financial_data_extractor import extract_financial_data_rag

# Set page config
st.set_page_config(
    page_title="Financial Data Extractor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        max-width: 100%;
    }
    .stDataFrame {
        width: 100%;
    }
    .section {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("📄 Finity Data Extractor")
    st.markdown("Upload financial documents (PDF/Excel/CSV) to extract structured financial data")

    # File upload
    uploaded_file = st.file_uploader(
        "Choose a financial document",
        type=["pdf", "csv", "xlsx", "xls"],
        accept_multiple_files=False
    )

    if uploaded_file is not None:
        # Save to temp file
        temp_path = f"./temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            with st.spinner("Analyzing document..."):
                result = extract_financial_data_rag(temp_path)

            # Show metadata
            with st.expander("Document Metadata", expanded=False):
                metadata = result.get("metadata", {})
                st.json(metadata, expanded=False)

            # Display financial data tables
            if result.get("financial_data"):
                st.subheader("📊 Extracted Financial Tables")
                for table in result["financial_data"]:
                    with st.expander(f"Table: {table.get('table_name', 'Unnamed Table')}"):
                        df = pd.DataFrame(table["rows"])
                        st.dataframe(df)
                        st.caption(f"Page: {table.get('page', 'N/A')}")

            # Display contextual text
            if result.get("contextual_text"):
                st.subheader("📝 Contextual Information")
                for text in result["contextual_text"]:
                    with st.expander(f"Contextual Text (Page {text.get('page', 'N/A')}", expanded=False):
                        st.markdown(f"```\n{text['content']}\n```")
                        st.write("Tags:", ", ".join(text.get("tags", [])))

            # Display notes
            if result.get("notes"):
                st.subheader("📌 Document Notes")
                for note in result["notes"]:
                    st.markdown(f"""
                    **Page {note.get('page', 'N/A')}**:
                    ```{note['content']}```
                    """)

            # Download button
            json_data = json.dumps(result, indent=2)
            st.download_button(
                label="Download Full Analysis (JSON)",
                data=json_data,
                file_name=f"{uploaded_file.name}_analysis.json",
                mime="application/json"
            )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
        finally:
            # Clean up temp file
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == "__main__":
    main()