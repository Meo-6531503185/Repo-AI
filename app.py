import streamlit as st

def main():
    st.set_page_config(page_title="Pdf Reader", page_icon=":books:")
    st.header=("A Place where you can ask anything about your Pdf")
    st.text_input("Ask anything")
    with st.sidebar:
        st.subheader("Your Documents")
        st.file_uploader("Upload your desired Pdfs")
        st.button("Process")

if __name__== "main":
    main()

