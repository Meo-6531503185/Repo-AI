@echo off
REM scripts\generate_and_test.bat

echo Generating test cases...
python python\ai_module.py

echo Running Maven tests...
cd java
mvn test
