# Talk to Law

## Disclaimer
This chatbot is a proof of concept and is not taken to be a serious legal advice tool. Any legal advice should be sought from a qualified legal professional. We do not accept any liability for any loss or damage incurred by use of this chatbot. Please use at your own risk.

## Description
This is a simple chatbot that can answer questions about the law. It is built using the LangChain framework. This project can be extended to any documents or digital domain.

## Installation
This project requires Python 3.7 or higher. It is recommended to use a virtual environment. To install the project, follow these steps:
1. Clone this repository
2. Install the requirements using `pip install -r requirements.txt`
3. Create a new file called `.env` and add the following:
```bash
COHERE_API_KEY=your_secret_api_key
OPENAI_API_KEY = your_secret_api_key
AI21_API_KEY = your_secret_api_key
JINACHAT_API_KEY = your_secret_api_key
```
Note: You need to only add those API keys which you will be using but COHERE_API_KEY is mandatory.

## Usage
1. Run `python legal_chat.py` to start the chatbot.
2. Select the chatbot you want to use.
3. Type in your question and press enter.
4. The chatbot will respond with an answer.
5. To exit, type `exit` and press enter.
