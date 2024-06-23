├── app.py                # Entry point for running the Flask app 

├── audio_processing.py   # Contains audio processing functions 

└── config.py             # Contains configuration settings

## Python Virtual Environment

virtual enviroment

1. 先python -m venv test_env(name)
2. list一下看看有没有安装
3. source test_env(name)/bin/activate 激活一下virtual environment
4. 推出ve deactivate

## Deploy On Server

**Create a Requirements File**: List all your dependencies in a `requirements.txt` file to ensure they can be easily installed on the server.

pip freeze > requirements.txt