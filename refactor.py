from langchain_google_vertexai import VertexAI
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st



#model
model = VertexAI(model="gemini-2.5-pro")

def refactor_code_with_ai(user_code, requirements, reasons):
   """Use Vertex AI to refine the user code based on test output."""
   system_template = f"""
   Refactor the following Java code to meet the requirements and address the issues and potential threats.
   Ensure that the refactored code adheres to best practices and satisfies the user requirements.
   Do not explain the changes made. Just refactor the code. Do not add test cases or file name.
   Provide only the raw Java code without any markdown or formatting symbols. Do not add the package name.
   ### Requirements ###
   {requirements}
   ### Reasons ###
   {reasons} are the test outputs why the previous AI-generated Java code did not pass the Maven tests.
   Refactor the code accordingly to satisfy these issues.
   """
   prompt_template = ChatPromptTemplate.from_messages([
       ("system", system_template),
       ("user", "{text}"),
       ("user", "{text2}")
   ])
   prompt = prompt_template.invoke({"requirements": requirements, "text": user_code, "text2": reasons})
   try:
       refined_code = model.invoke(prompt)
       print("AI successfully refined the code.")
    #    return f"\n{refined_code}\n"
       return refined_code
   except Exception as e:
       print(f"An error occurred while calling Vertex AI: {e}")
       return user_code
   
def Overview_of_repository(user_question):
   # st.write("Overview being called...")


   repo_data = st.session_state.repo_data


   # Inspect the repo_data structure
   # st.write("repo_data contents:", repo_data)


   # Ensure that repo_data contains files
   if not repo_data:
       st.write("Error: No files found in repo_data")
       return None


   # Extract file names and their contents
   file_contents = []
   for filename, content in repo_data.items():
       file_contents.append(f"File: {filename}\n{content}\n\n")  # Adding filename for clarity


   if not file_contents:
       st.write("No files were found in repo_data.")
       return None
  
   # st.write("File contents:", file_contents)


   # Escape the curly braces in the system template
   system_template = f"""
   Analyze the structure of the given GitHub repository files, which contains the following files and modules:
   {{file_contents}}


   Based on this information, please provide a clear, high-level explanation of the repository.
   Identify the purpose of the project, its key modules, and dependencies.
   Describe the main components, such as entity classes, services, controllers, and utility functions.
   Summarize how different parts of the codebase interact, highlighting any important frameworks or technologies used (e.g., Spring Boot, Hibernate, REST APIs).
   Keep the explanation concise and developer-friendly.
   """




   # Create the prompt with user input and system context
   prompt_template = ChatPromptTemplate.from_messages(
       [("system", system_template), ("user", "{text}")]
   )


   # Prepare the prompt with the user's question
   prompt_input = prompt_template.invoke({"file_contents":file_contents,"text": user_question})


   try:
       # Assuming 'model' is the language model you're using to get the response
       explanation = model.invoke(prompt_input)
       # st.write("Model Response:\n", explanation)  # Check the model's response
       return explanation
   except Exception as e:
       st.write(f"An error occurred while processing the output: {e}")
       return None




      
def Enhance_full_project(user_question):
   repo_data = st.session_state.repo_data
   if not repo_data:
       st.write("Error: No files found in repo_data")
       return None
   file_contents = []
   for filename, content in repo_data.items():
       file_contents.append(f"File: {filename}\n{content}\n\n")  # Adding filename for clarity


   if not file_contents:
       st.write("No files were found in repo_data.")
       return None
  
  
   system_template = f"""
   Perform a comprehensive enhancement of the entire project {{file_contents}} by analyzing and refactoring code while maintaining all existing functionality. Follow a structured approach:*


1️ Codebase Analysis:
Identify code duplication, inefficiencies, and areas that violate best practices.
Ensure consistent formatting, proper naming conventions, and modularization.
2️ Dependency & Configuration Improvements:
Optimize pom.xml (remove unused dependencies, update outdated libraries).
Improve project configurations (application.properties, environment settings).
3️ Performance Enhancements:
Optimize database queries, caching strategies, and avoid redundant computations.
Improve asynchronous processing where applicable.
4️ Architecture & Design Patterns:


Ensure the project follows clean architecture principles (e.g., layered architecture, microservices).


Refactor tightly coupled classes to improve maintainability.


5️ Testing & Validation:


Ensure all modifications pass unit tests, integration tests, and build verification (mvn test, mvn verify).


If test coverage is low, suggest or generate missing test cases.


6️ Final Verification & Recommendations:


Summarize all enhancements and their impact.


Ensure the project builds successfully and all improvements align with best practices.


Return a structured summary of enhancements and confirm that the project remains functional after modifications.
   """
   prompt_template = ChatPromptTemplate.from_messages(
       [("system", system_template), ("user", "{text}")]
   )
   prompt_input = prompt_template.invoke({"file_contents":file_contents,"text": user_question})
  
   try:
       explanation = model.invoke(prompt_input)
       #st.write(explanation)
       return explanation
      
   except Exception as e:
       print(f"An error occurred while processing test output: {e}")