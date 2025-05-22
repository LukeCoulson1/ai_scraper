@echo off
REM Ensure required packages are installed (optional, but helpful)
pip install streamlit pandas selenium beautifulsoup4 tqdm langdetect llama-cpp-python openpyxl

REM Run the Streamlit app
streamlit run webui.py

pause