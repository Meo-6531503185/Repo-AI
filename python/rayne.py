import os
import subprocess
from langchain_google_vertexai import VertexAI
from langchain_core.prompts import ChatPromptTemplate

model = VertexAI(model = "gemini-1.5-flash-002")
def generate_java_code_and_test(user_code, requirements):
    """Generate Java source and test files based on user input."""
    java_code = user_code
    with open("java/src/main/java/GeneratedCode.java", "w") as file:
        file.write(java_code)
    
    system_template = f"""You are a proficient Java developer and an expert in Test-Driven development. 
    Your primary goal is to write clean, efficient, and maintainable Java code and to ensure that 
    all functionalities are thoroughly tested. Here are the requirements for the code you need to write: {requirements}"""
    prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", {"code"})]
    
    
)
    prompt = prompt_template.invoke({"requirements": requirements,"code": user_code})
    test_case =  model.invoke(prompt)
    with open("java/src/test/java/TestGeneratedCode.java", "w") as file:
        file.write(test_case.content)


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
    PROJECT_ID = "langchain-iris-chatbot"
    LOCATION = "us-central1"
    MODEL_NAME = "gemini-1.5-flash-002"  # Use the correct Vertex AI model name

    aiplatform.init(project=PROJECT_ID, location=LOCATION)

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

    Provide the updated Java code only and include explanations for the changes.
    """

    try:
        model = aiplatform.Model(model_name="gemini-1.5-flash-002")
        response = model.predict(
            content=prompt,
            parameters={
                "temperature": 0.5,
                "maxOutputTokens": 600,
                "topK": 40,
                "topP": 0.9,
            },
        )

        refined_code = response.predictions[0]["content"]
        print("AI successfully refined the code.")
        return refined_code

    except Exception as e:
        print(f"An error occurred while calling Vertex AI: {e}")
        return user_code


def handle_refactor_and_test(user_code, requirements):
    """Main workflow for refactoring and testing."""
    test_output = ""  # Initialize test_output
    generate_java_code_and_test(user_code, requirements)

    while True:
        test_output, test_status = run_maven_tests()

        if test_status == 0:
            print("All tests passed!")
            break
        else:
            print("Tests failed. Refactoring and retrying...")
            user_code = refactor_code_with_ai(user_code, requirements, test_output)


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
