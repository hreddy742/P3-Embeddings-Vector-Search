import streamlit as st

st.title("File Upload Test")

# file_uploader shows a drag and drop area in the browser
# type=["pdf"] means only PDF files are accepted
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# uploaded_file is None until user uploads something
if uploaded_file is not None:
    st.write(f"Filename : {uploaded_file.name}")
    st.write(f"File size: {uploaded_file.size} bytes")