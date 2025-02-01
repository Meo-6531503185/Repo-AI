import subprocess
import os
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
from google.cloud import aiplatform
import vertexai
from vertexai.language_models import CodeGenerationModel



# def generate_java_code_and_test(user_code, requirements):
#     """Generate Java source and test files based on user input."""
#     # Generate the Java file
#     java_code = f"""
#     public class GeneratedCode {{
#         public int someMethod() {{
#             return 42;  // Example implementation
#         }}
#     }}
#     """
#     with open("java/src/main/java/GeneratedCode.java", "w") as file:
#         file.write(java_code)

#     # Generate the corresponding JUnit test file
#     test_case = f"""
#     import static org.junit.jupiter.api.Assertions.*;
#     import org.junit.jupiter.api.Test;

#     public class TestGeneratedCode {{
#         @Test
#         public void testSomeMethod() {{
#             assertEquals(42, new GeneratedCode().someMethod(), "Test failed for someMethod");
#         }}
#     }}
#     """
#     with open("java/src/test/java/TestGeneratedCode.java", "w") as file:
#         file.write(test_case)


# def run_maven_tests():
#     """Run Maven tests and return the output."""
#     maven_path = "C:\\Program Files\\apache-maven-3.9.5\\bin\\mvn.cmd"
#     try:
#         result = subprocess.run([maven_path, "test"], cwd="java", capture_output=True, text=True, check=True)
#         return result.stdout, 0
#     except subprocess.CalledProcessError as e:
#         return e.stdout, e.returncode


# # def refactor_code_with_ai(user_code, requirements):
# #     """Placeholder for AI-based refactoring logic."""
# #     # Here, you can integrate an AI model like OpenAI's Codex or Vertex AI
# #     # Pass the `user_code` and `test_output` for analysis and refinement
    
# #     print("Refactoring code with AI based on failed tests...")
# #     return user_code  # Placeholder - refine this logic

# def refactor_code_with_ai(user_code, requirements):
#     """Use Vertex AI to refine the user code based on test output."""

#     # Initialize Vertex AI
#     PROJECT_ID = "langchain-iris-chatbot"  # Replace with your GCP project ID
#     LOCATION = "us-central1"  # Adjust if your Vertex AI region differs
#     MODEL_NAME = "gemini-1.5-flash-002"  # Replace with the appropriate model name

#     aiplatform.init(project=PROJECT_ID, location=LOCATION)

#     # Prepare the prompt for the AI model
#     prompt = f"""
#     Refactor the following Java code to meet the requirements and address the issues identified in the test output.
#     Ensure that the refactored code adheres to best practices and satisfies the user requirements.

#     ### User Code ###
#     {user_code}

#     ### Requirements ###
#     {requirements}

#     ### Test Output ###
#     {test_output}

#     Provide the updated Java code only. And include explanations and why you update this way.
#     """

#     # Request refinement from the Vertex AI model
#     try:
#         response = aiplatform.TextGenerationModel.from_pretrained(MODEL_NAME).predict(
#             content=prompt,
#             temperature=0.5,  # Adjust for creativity level
#             max_output_tokens=600,  # Increase if your code is longer
#             top_k=40,
#             top_p=0.9,
#         )

#         # Extract the generated Java code
#         refined_code = response.text

#         print("AI successfully refined the code.")
#         return refined_code

#     except Exception as e:
#         print(f"An error occurred while calling Vertex AI: {e}")
#         return user_code  # Return the original code in case of an error


# def handle_refactor_and_test(user_code, requirements):
#     """Main workflow for refactoring and testing."""
#     # Step 1: Generate Java code and tests
#     refined_code= refactor_code_with_ai(user_code, requirements)
#     generate_java_code_and_test(refined_code, requirements)

#     # Step 2: Run Maven tests
#     test_output, test_status = run_maven_tests()

#     # Step 3: Handle test results
#     if test_status == 0:
#         print("All tests passed!")
#     else:
#         print("Tests failed. Refactoring and retrying...")
#         # Refactor code with AI and retry
#         refined_code = refactor_code_with_ai(user_code, requirements)
#         handle_refactor_and_test(refined_code, requirements)

# def main():
#     """Streamlit UI for AI and Maven workflow."""
#     st.title("AI-Powered Java Refactoring and Testing")
#     user_code = st.text_area("Enter the initial user code or requirements")
#     requirements = st.text_area("Enter the user requirements")

#     if st.button("Refactor and Test"):
#         if not user_code or not requirements:
#             st.error("Please provide both code and requirements!")
#         else:
#             refined_code = refactor_code_with_ai(user_code, requirements)
#             handle_refactor_and_test(refined_code, requirements)

    

# if __name__ == "__main__":
#     main()

def generate_java_code_and_test(user_code, requirements):
    """Generate Java source and test files based on user input."""
    # Write the Java source code to a file
    java_code = f"""
    public class GeneratedCode {{
        public int someMethod() {{
            return 42;  // Example implementation
        }}
    }}
    """
    with open("java/src/main/java/GeneratedCode.java", "w") as file:
        file.write(java_code)

    # Write a corresponding JUnit test case based on requirements
    test_case = f"""
    import static org.junit.jupiter.api.Assertions.*;
    import org.junit.jupiter.api.Test;

    public class TestGeneratedCode {{
        @Test
        public void testSomeMethod() {{
            GeneratedCode gc = new GeneratedCode();
            assertEquals(10, gc.someMethod(), "The method did not return the expected value.");
        }}
    }}
    """
    with open("java/src/test/java/TestGeneratedCode.java", "w") as file:
        file.write(test_case)


def run_maven_tests():
    """Run Maven tests and return the output."""
    maven_path = "C:\\Program Files\\apache-maven-3.9.5\\bin\\mvn.cmd"
    try:
        result = subprocess.run([maven_path, "test"], cwd="java", capture_output=True, text=True, check=True)
        return result.stdout, 0
    except subprocess.CalledProcessError as e:
        return e.stdout, e.returncode


def refactor_code_with_ai(user_code, requirements, test_output):
    """Use Vertex AI to refine the user code based on test output."""

    # Initialize Vertex AI
    PROJECT_ID = "langchain-iris-chatbot"  # Replace with your GCP project ID
    LOCATION = "us-central1"  # Adjust if your Vertex AI region differs
    # MODEL_NAME = "gemini-1.5-flash-002"  # Replace with the appropriate model name

    vertexai.init(project=PROJECT_ID, location=LOCATION)

    # Prepare the prompt for the AI model
    prompt = f"""
    Refactor the following Java code to meet the requirements and address the issues identified in the test output.
    Ensure that the refactored code adheres to best practices and satisfies the user requirements.

    ### User Code ###
    {user_code}

    ### Requirements ###
    {requirements}

    ### Test Output ###
    {test_output}

    Provide the updated Java code only. Explain the changes you made and why.
    """

    # Request refinement from the Vertex AI model
    try:
        parameters = {
            "temperature": 0.2,  # Temperature controls the degree of randomness in token selection.
            "max_output_tokens": 64,  # Token limit determines the maximum amount of text output.
        }

        code_completion_model = CodeGenerationModel.from_pretrained("code-gecko@001")
        response = code_completion_model.predict(
        prefix="def reverse_string(s):", **parameters
        )

        # generative_multimodal_model = GenerativeModel("gemini-1.5-flash-002")
        # # response = generative_multimodal_model.generate_content(["What is shown in this image?", image])
        # response = generative_multimodal_model._generate_content(
        #     content=prompt,
        #     temperature=0.5,  # Adjust for creativity level
        #     max_output_tokens=600,  # Increase if your code is longer
        #     top_k=40,
        #     top_p=0.9,
        # )

        # Extract the generated Java code
        refined_code = response.text
        print("AI successfully refined the code.")
        return refined_code

    except Exception as e:
        print(f"An error occurred while calling Vertex AI: {e}")
        return user_code  # Return the original code in case of an error


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

        # Refactor code with AI and retry
        refined_code = refactor_code_with_ai(user_code, requirements, test_output)
        handle_refactor_and_test(refined_code, requirements)


def main():
    """Streamlit UI for AI and Maven workflow."""
    st.title("AI-Powered Java Refactoring and Testing")

    user_code = st.text_area("Enter the initial Java code")
    requirements = st.text_area("Enter the user requirements")

    if st.button("Refactor and Test"):
        if not user_code or not requirements:
            st.error("Please provide both code and requirements!")
        else:
            handle_refactor_and_test(user_code, requirements)


if __name__ == "__main__":
    main()

