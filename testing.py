import re
from langchain_google_vertexai import VertexAI
import os
from langchain_core.prompts import ChatPromptTemplate
import subprocess
import streamlit as st
from refactor import *

def extract_class_name(java_code):
   """Extract the first class name from Java code."""
   match = re.search(r'\bclass\s+(\w+)', java_code)
   return match.group(1) if match else "GeneratedCode"


#model
model = VertexAI(model="gemini-2.5-pro")


def generate_java_code_and_test(user_code):
   """Generate Java source and test files based on user input."""
   output_dir = "../Repo-AI/java/src/main/java/"
   os.makedirs(output_dir, exist_ok=True)
   class_name = extract_class_name(user_code)
   with open(os.path.join(output_dir, f"{class_name}.java"), "w") as file:
       file.write(user_code)
       
   system_template = f"""Generate JUnit test cases for the following user's Java program based on the official guidelines from https://junit.org/junit5/docs/current/user-guide/#writing-tests.
Ensure the test cases cover valid inputs, invalid inputs, and edge cases.
Follow these best practices:
- **Do NOT call private methods directly**; instead, test them indirectly through **public methods** such as getters and setters.
- Use **assertions** to verify expected behavior.
- Focus on **state validation** rather than implementation details.
- If necessary, use setter methods to modify the object state before calling the tested methods.
- Ensure edge cases, such as extreme values, null inputs (if applicable), and boundary conditions, are tested.
Provide only the raw Java code without any markdown or formatting symbols. 
Do NOT add the package name and class name at the top of the code.
"""
   prompt_template = ChatPromptTemplate.from_messages([
       ("system", system_template),
       ("user", "{text}")
   ])
    
   prompt = prompt_template.invoke({"text": user_code})
   test_case = model.invoke(prompt)  # Generate test case
    
   test_class_name = extract_class_name(test_case)
   test_output_dir = "../Repo-AI/java/src/test/java/"
   os.makedirs(test_output_dir, exist_ok=True)

   with open(os.path.join(test_output_dir, f"{test_class_name}.java"), "w") as file:
        file.write(test_case)
    
   return test_case  # Return the test 
      
  
def run_maven_tests():
   """Run Maven tests and return the output."""
   maven_path = "C:\\Program Files\\apache-maven-3.9.5\\bin\\mvn.cmd"
   project_path = r"C:\\Users\\user\\Documents\\RepoAI\\Repo-AI\\java"
   try:
       result = subprocess.run([maven_path, "test"], cwd=project_path, capture_output=True, text=True, check=True)
       return result.stdout, 0
   except subprocess.CalledProcessError as e:
       return e.stdout, e.returncode
   
def analyze_test_failures(test_output, user_code, requirements):
   system_template = """
   Analyze the following Maven test output and determine the cause of failure.
   Explain why the test failed and classify whether the issue is in the class implementation or the test cases.
   Use logical reasoning rather than relying on specific keywords. Identify:
   - Whether the failure is due to missing methods, incorrect method signatures, or unexpected exceptions (likely requiring class refactoring).
   - Whether the failure is due to incorrect test expectations, improper assertions, or flawed test logic (likely requiring test case refactoring).
   - If the failure is ambiguous, suggest manual review.
   Based on your reasoning, determine the appropriate action:
   - If the class is incorrect, return "Regenerate Class File".
   - If the test cases need fixing, return "Regenerate Test Cases".
   - If it is unclear, return "Manual Review Needed".
   Provide a clear and structured explanation before your decision.
   """
   prompt_template = ChatPromptTemplate.from_messages(
       [("system", system_template), ("user", "{text}")]
   )
   prompt = prompt_template.invoke({"text": test_output})
   try:
       explanation = model.invoke(prompt)
       # Automatically trigger refactoring based on AI's analysis
       if "Regenerate Class File" in explanation:
           refined_code = refactor_code_with_ai(user_code, requirements, test_output)
           st.write("Class file has some issues. Class file has been regenerated. Running new tests...")
           generate_java_code_and_test(refined_code)
       elif "Regenerate Test Cases" in explanation:
           st.write("Test cases have issues. Generating updated test cases...")
           generate_java_code_and_test(user_code)
       else:
           st.write("Manual review is needed.")
   except Exception as e:
       print(f"An error occurred while processing test output: {e}")
      
  
  