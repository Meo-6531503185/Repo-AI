import subprocess
import re
from difflib import SequenceMatcher
import xml.etree.ElementTree as ET
import os
import csv

# --- Run Maven Tests and Parse Output ---
def run_maven_tests(project_path):
    maven_path = "C:\\Program Files\\apache-maven-3.9.5\\bin\\mvn.cmd"
    result = subprocess.run(
        [maven_path, 'test'],
        cwd=project_path,
        capture_output=True,
        text=True
    )
    output = result.stdout
    summary = re.search(r"Tests run: (\d+), Failures: (\d+), Errors: (\d+), Skipped: (\d+)", output)

    if summary:
        total = int(summary.group(1))
        failures = int(summary.group(2))
        errors = int(summary.group(3))
        skipped = int(summary.group(4))
        passed = total - failures - errors - skipped
        test_accuracy = (passed / total) * 100 if total else 0
        return test_accuracy, passed, total
    else:
        return 0, 0, 0

# --- Parse Checkstyle XML for Violations ---
def parse_checkstyle_violations(report_path):
    if not os.path.exists(report_path):
        return 0
    tree = ET.parse(report_path)
    root = tree.getroot()
    violations = sum(len(file.findall('error')) for file in root.findall('file'))
    return violations

# --- Calculate Static Code Quality Score ---
def static_code_quality_score(violations):
    return max(0, 100 - (violations * 2))  # Each violation costs 2 points

# --- Retry Penalty Score ---
def retry_penalty_score(num_retries):
    return max(0, 100 - (num_retries * 20))  # Each retry costs 20 points

# # --- Semantic Similarity Score ---
# def similarity_score(original_code, ai_code):
#     ratio = SequenceMatcher(None, original_code, ai_code).ratio()
#     return ratio * 100

# --- Placeholder for Refactoring Type Accuracy ---
def refactor_type_score():
    # Future improvement: classify refactor types using AST diff or AI annotations
    return 90

# --- Final ARAS Computation ---
def compute_aras(
    test_score,
    quality_score,
    retry_score,
    similarity_score=0,
    refactor_type_score=0
):
    weights = {
        "test": 0.4,
        "quality": 0.25,
        "retry": 0.15,
        "similarity": 0.1,
        "refactor": 0.1
    }

    score = (
        test_score * weights["test"] +
        quality_score * weights["quality"] +
        retry_score * weights["retry"] +
        similarity_score * weights["similarity"] +
        refactor_type_score * weights["refactor"]
    )

    return round(score, 2)

# --- Save Results to CSV ---
def log_to_csv(data, csv_path='aras_results.csv'):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(data.keys())
        writer.writerow(data.values())

# --- Main Evaluation Function ---
def evaluate_refactoring(project_path, checkstyle_report_path, retries=0):
    # Get Maven Test Accuracy
    test_score, passed, total = run_maven_tests(project_path)
    
    # Parse Checkstyle Report for Violations
    violations = parse_checkstyle_violations(checkstyle_report_path)
    quality = static_code_quality_score(violations)
    
    # Retry Score
    retry_score = retry_penalty_score(retries)
    
    # # Similarity Score (if both original and refactored code are provided)
    # similarity = similarity_score(original_code, refactored_code) if original_code and refactored_code else 0
    
    # Refactor Type Score (Placeholder)
    refactor_score = refactor_type_score()

    # Compute Final ARAS Score
    aras = compute_aras(test_score, quality, retry_score, refactor_score)

    result = {
        "Test Score": test_score,
        "Code Quality Score": quality,
        "Retry Score": retry_score,
        # "Similarity Score": similarity,
        "Refactor Type Score": refactor_score,
        "ARAS": aras,
        "Tests Passed": passed,
        "Total Tests": total,
        "Violations": violations,
        "Retries": retries
    }

    # Log the results to CSV
    log_to_csv(result)
    
    return result

results = evaluate_refactoring(
    project_path=r"C:\\Users\\user\\Documents\\RepoAI_Github\\Repo-AI\\java",
    checkstyle_report_path=r"C:\\Users\\user\\Documents\\RepoAI_Github\\Repo-AI\\java\\target\\checkstyle-result.xml",
    retries=1,
    # original_code="public class A { void foo() {} }",
    # refactored_code="public class A { void bar() {} }"
)

print("ARAS Score:", results["ARAS"])

