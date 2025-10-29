from langchain_google_vertexai import VertexAI
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

class OverviewAgent:
    def __init__(self):
        self.model = VertexAI(model_name="gemini-2.5-flash")
    def run(self, user_question: str):
        repo_data = st.session_state.get("repo_data", {})
        if not repo_data: return "⚠️ No repository data available."
        files_summary = "\n".join([f"File: {f}" for f in repo_data.keys()])
        system_template = "You are an expert code analyst. Explain what the repository is about."
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("user", "User Question: {question}\n\nRepository Files:\n{files}")
        ]).invoke({"question": user_question, "files": files_summary})
        return self.model.invoke(prompt)

class MultiFileRefactorAgent:
    def __init__(self):
        self.model = VertexAI(model_name="gemini-2.5-flash")
    def run(self, user_question: str, repo_data: dict):
        results = {}
        for f, content in repo_data.items():
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Refactor the following Java code for performance, readability, and security."),
                ("user", "User Request: {question}\n\nCode:\n{code}")
            ]).invoke({"question": user_question, "code": content})
            results[f] = self.model.invoke(prompt)
        return results

# class FeatureAgent:
#     def __init__(self):
#         self.model = VertexAI(model_name="gemini-2.5-flash")
#     def run(self, user_question: str, code_context: str):
#         prompt = ChatPromptTemplate.from_messages([
#             ("system", "Implement a new feature based on user request."),
#             ("user", "User Request: {question}\n\nExisting Code:\n{code}")
#         ]).invoke({"question": user_question, "code": code_context})
#         return self.model.invoke(prompt)

# class MultiFileFeatureAgent:
#     def __init__(self):
#         self.feature_agent = FeatureAgent()
#     def run(self, user_question: str, repo_data: dict, target_file: str):
#         impacted_files = [target_file]
#         target_class = target_file.split(".")[0]
#         for f, content in repo_data.items():
#             if f != target_file and target_class in content:
#                 impacted_files.append(f)
#         results = {}
#         for f in impacted_files:
#             results[f] = self.feature_agent.run(user_question, repo_data.get(f, ""))
#         return results

class EnhanceAgent:
    def __init__(self):
        self.model = VertexAI(model_name="gemini-2.5-flash")
    def run(self, user_question: str):
        repo_data = st.session_state.get("repo_data", {})
        if not repo_data: return "⚠️ No repository data available."
        files = "\n".join(repo_data.keys())
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Perform comprehensive project enhancement."),
            ("user", "User Request: {question}\n\nFiles:\n{files}")
        ]).invoke({"question": user_question, "files": files})
        return self.model.invoke(prompt)
