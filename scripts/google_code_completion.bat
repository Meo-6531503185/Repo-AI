@echo off

:: Set variables
set MODEL_ID=code-gecko
set PROJECT_ID=YOUR_PROJECT_ID

:: Execute the curl command
curl ^
-X POST ^
-H "Authorization: Bearer $(gcloud auth print-access-token)" ^
-H "Content-Type: application/json" ^
https://us-central1-aiplatform.googleapis.com/v1/projects/%PROJECT_ID%/locations/us-central1/publishers/google/models/%MODEL_ID%:predict -d ^
$"{
  \"instances\": [
    {
      \"prefix\": \"def reverse_string(s):\",
      \"suffix\": \"\"
    }
  ],
  \"parameters\": {
    \"temperature\": 0.2,
    \"maxOutputTokens\": 64,
    \"candidateCount\": 1
  }}
"
