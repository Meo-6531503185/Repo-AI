import os
import subprocess
import streamlit as st
from git import Repo
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
import vertexai
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.auth 
from langchain_google_vertexai import VertexAI
from langchain_google_vertexai import VertexAIEmbeddings
# from langchain_chroma import Chroma
from urllib.parse import urlparse 
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
# from htmlTemplates import css, bot_template, user_template
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
# Test Data
def generate_java_code_and_test(user_code, requirements):
    """Generate Java source and test files based on user input."""
    # Generate the Java file
    java_code = f"""
    public class GeneratedCode {{
        public int someMethod() {{
            return 42;  // Example implementation
        }}
    }}
    """
    with open("java/src/main/java/GeneratedCode.java", "w") as file:
        file.write(java_code)

    # Generate the corresponding JUnit test file
    test_case = f"""
    import static org.junit.jupiter.api.Assertions.*;
    import org.junit.jupiter.api.Test;

    public class TestGeneratedCode {{
        @Test
        public void testSomeMethod() {{
            assertEquals(42, new GeneratedCode().someMethod(), "Test failed for someMethod");
        }}
    }}
    """
    with open("java/src/test/java/TestGeneratedCode.java", "w") as file:
        file.write(test_case)


def run_maven_tests():
    """Run Maven tests and return the output."""
    maven_path = "C:\\Program Files\\apache-maven-3.9.5\\bin\\mvn.cmd"
    result = subprocess.run([maven_path, "test"], cwd="java", capture_output=True, text=True)
    return result.stdout, result.returncode


def handle_refactor_and_test(user_code, requirements):
    """Main workflow for refactoring and testing."""
    # Step 1: Generate Java code and tests
    generate_java_code_and_test(user_code, requirements)

    # Step 2: Run Maven tests
    test_output, test_status = run_maven_tests()

    # Step 3: Handle test results
    if test_status == 0:
        st.success("All tests passed!")
    else:
        st.error("Tests failed. Refactoring and retrying...")
        # Here, you can invoke AI to refine the code further
        # Example: Call VertexAI to improve the Java implementation


def main():
    """Streamlit UI for AI and Maven workflow."""
    st.title("AI-Powered Java Refactoring and Testing")
    user_code = st.text_area("Enter the initial user code or requirements")
    requirements = st.text_area("Enter the user requirements")

    if st.button("Refactor and Test"):
        if not user_code or not requirements:
            st.error("Please provide both code and requirements!")
        else:
            handle_refactor_and_test(user_code, requirements)




if __name__ == "__main__":
    main()

