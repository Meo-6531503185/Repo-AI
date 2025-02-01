import streamlit as st
import os
import subprocess
from langchain_google_vertexai import VertexAI
from langchain_core.prompts import ChatPromptTemplate
import re

def extract_class_name(java_code):
    """Extract the first class name from Java code."""
    match = re.search(r'\bclass\s+(\w+)', java_code)
    return match.group(1) if match else "GeneratedCode"

model = VertexAI(model = "gemini-1.5-flash-002")


def generate_java_code_and_test(user_code, requirements):
    """Generate Java source and test files based on user input."""
    output_dir = "../java/src/main/java/"
    os.makedirs(output_dir, exist_ok=True)  # Create directories if they don't exist

    # java_code = f"""
    # public class GeneratedCode {{
    #     public int someMethod() {{
    #         return 42;  // Example implementation
    #     }}
    # }}
    # """
    java_code = user_code
    class_name = extract_class_name(java_code)
    
    # with open("java/src/main/java/GeneratedCode.java", "w") as file:
    #     file.write(java_code)
    with open(os.path.join(output_dir, f"{class_name}.java"), "w") as file:
        file.write(java_code)
    
    system_template = f"""You are a proficient Java developer and an expert in Test-Driven development. 
    Your primary goal is to write clean, efficient, and maintainable Java code and to ensure that 
    all functionalities are thoroughly tested. Here are the requirements for the code you need to write: {requirements}.
    Reminders : 
    1. Just create the tests. Do not explain anything.
    2. Do NOT include ```java or any other unnecessary markers in the generated code.
    3. Ensure the test filename follows the pattern: <ClassName>Test.java.
    4. Do not include the user code. Just write the test code."""
    prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
    
     
    )
    prompt = prompt_template.invoke({"requirements": requirements,"text": user_code})
    test_case =  model.invoke(prompt)
    class_name = extract_class_name(test_case)
    output_dir = "../java/src/test/java/"
    os.makedirs(output_dir, exist_ok=True)  # Create directories if they don't exist

    with open(os.path.join(output_dir,  f"{class_name}.java"), "w") as file:
        file.write(test_case)

        
def run_maven_tests():
    """Run Maven tests and return the output."""
    maven_path = "C:\\Program Files\\apache-maven-3.9.5\\bin\\mvn.cmd"
    try:
        result = subprocess.run([maven_path, "test"], cwd="java", capture_output=True, text=True, check=True)
        return result.stdout, 0
    except subprocess.CalledProcessError as e:
        return e.stdout, e.returncode
    
def refactor_code_with_ai(user_code, requirements):
    """Use Vertex AI to refine the user code based on test output."""
    

    # Prepare the prompt for the AI model
    system_template = f"""
    Refactor the following Java code to meet the requirements and address the issues and potential threats.
    Ensure that the refactored code adheres to best practices and satisfies the user requirements. Do not explain the changes made. Just refactor the code.

    
    ### Requirements ###
    {requirements}

    

    Provide the updated Java code only and include explanations for the changes.
    """
    prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
    
    
    )
    prompt = prompt_template.invoke({"requirements": requirements,"text": user_code})
     
    
    
    try:
        refined_code = model.invoke(prompt)

        
        print("AI successfully refined the code.")
        return refined_code
        

    except Exception as e:
        print(f"An error occurred while calling Vertex AI: {e}")
        return user_code
    
def handle_refactor_and_test(user_code, requirements):
    """Main workflow for refactoring and testing."""
    # test_output = ""  # Initialize test_output
    refined_code = refactor_code_with_ai(user_code, requirements)
    generate_java_code_and_test(refined_code, requirements)

    while True:
        test_status = run_maven_tests()

        if test_status == 0:
            print("All tests passed!")
            break
        else:
            print("Tests failed. Refactoring and retrying...")
            refined_code = refactor_code_with_ai(user_code, requirements)
            generate_java_code_and_test(refined_code, requirements)




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


