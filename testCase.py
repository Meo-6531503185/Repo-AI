# testCase.py
from app import log_test_case
from testData import get_test_data
# Sample test cases
test_data = [
    {
        "question": "What is the purpose of this repository?",
        "expected_answer": "This repository is about portfolio website built by using React Framework."
    },
]
def calculate_accuracy_and_precision():
    correct_answers = 0
    total_questions = len(test_data)
    precision_numerator = 0
    precision_denominator = 0
    # Iterate over the test cases
    for case in test_data:
        question = case["question"]
        expected_answer = case["expected_answer"]
        # Simulate a response from the chatbot (mocked for now)
        response = "This repository is about portfolio website built by using React Framework."
        # Log the test case and store the response
        result = log_test_case(question, response, expected_answer)
        # Accuracy: If the response matches the expected answer exactly, it's considered correct
        if result["match"]:
            correct_answers += 1
        # Precision: Check if the answer is correct (based on some partial matching or fuzzy logic)
        if expected_answer in result["response"]:
            precision_numerator += 1
        if result["response"]:
            precision_denominator += 1
        # Print real-time result of the current test case
        print(f"Question: {question}")
        print(f"Response: {result['response']}")
        print(f"Expected Answer: {expected_answer}")
        print(f"Match: {'Yes' if result['match'] else 'No'}")
        print("-" * 50)
    # Calculate accuracy and precision
    accuracy = correct_answers / total_questions
    precision = precision_numerator / precision_denominator if precision_denominator > 0 else 0
    return accuracy, precision
# Example of running the test
if __name__ == "__main__":
    accuracy, precision = calculate_accuracy_and_precision()
    # Display final accuracy and precision in real-time
    print(f"Final Accuracy: {accuracy * 100:.2f}%")
    print(f"Final Precision: {precision * 100:.2f}%")
