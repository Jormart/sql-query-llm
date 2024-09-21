# sql_translator.py

import os
import logging

from dotenv import load_dotenv

from llm_utils import select_llm
from langchain.schema import SystemMessage, HumanMessage

# Load environment variables from the .env file
load_dotenv()

# Setup logging to capture events and useful information
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


def generate_prompt_template(sql_query):
    """
    Generate a natural language prompt based on the SQL query with special instructions.

    Args:
        sql_query (str): The SQL query to translate.

    Returns:
        str: The prompt template.
    """
    template = f"""
Convert the following SQL query into a natural language question:

SQL Query:
{sql_query}

Instructions:
- Do not mention any technical terms like data, tables, or databases.
- Make the question clear and concise.
- Simplify the language where possible.
- Assume geographical context if applicable.
"""
    return template.strip()


def translate_to_human(sql_query, model_name):
    """
    Translate a SQL query into a natural language question using the selected LLM.

    Args:
        sql_query (str): The SQL query to translate.
        model_name (str): The name of the LLM model to use.

    Returns:
        str: The translated natural language question.
    """
    llm = select_llm(model_name)
    if llm is None:
        logging.error("Invalid LLM model selected.")
        return "Error: Invalid LLM model selected."

    prompt_text = generate_prompt_template(sql_query)

    # Create messages using SystemMessage and HumanMessage
    messages = [
        SystemMessage(content="You are an expert in translating SQL queries into natural language questions."),
        HumanMessage(content=prompt_text)
    ]

    try:
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        logging.error(f"Error translating query to natural language: {e}")
        return f"Error in translation process: {e}"
