# llm_utils.py

import os
import sqlite3
import logging
import re
import ast

from dotenv import load_dotenv
from fuzzywuzzy import fuzz

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage

# Load environment variables from the .env file
load_dotenv()

# Read the API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Setup logging to capture events and useful information
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Supported models
OPENAI_MODELS = ["gpt-3.5-turbo", "gpt-4o-mini"]
GROQ_MODELS = ["llama-3.1-8b-instant", "llama-3.1-70b-versatile"]


def select_llm(model_name: str):
    """
    Select the appropriate LLM based on the model name.

    Args:
        model_name (str): Name of the LLM model.

    Returns:
        LLM instance or None if invalid model name.
    """
    if model_name in GROQ_MODELS:
        if not GROQ_API_KEY:
            logging.error("GROQ API key is missing.")
            return None
        return ChatGroq(model=model_name, temperature=0)
    elif model_name in OPENAI_MODELS:
        if not OPENAI_API_KEY:
            logging.error("OpenAI API key is missing.")
            return None
        # Ensure that ChatOpenAI is instantiated correctly
        return ChatOpenAI(model_name=model_name, temperature=0)
    else:
        logging.error(f"Invalid model name: {model_name}")
        return None


def get_llm_answer_direct(question: str, model_name: str) -> str:
    """
    Generate an answer using the LLM directly based on the question.

    Args:
        question (str): The user's original question.
        model_name (str): The name of the LLM model to use.

    Returns:
        str: The LLM-generated answer to the question.
    """
    llm = select_llm(model_name)
    if llm is None:
        logging.error("Invalid LLM model selected.")
        return None

    prompt = f"""
You are a helpful assistant. Answer the following question directly.

Question:
{question}

Answer:
"""

    try:
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=prompt)
        ]
        response = llm.invoke(messages)
        question_answer = response.content.strip()
        return question_answer
    except Exception as e:
        logging.error(f"Error invoking LLM for direct question: {e}")
        return None


def execute_query(sqlite_file: str, query: str):
    """
    Execute a SQL query on a SQLite database.

    Args:
        sqlite_file (str): Path to the SQLite database file.
        query (str): The SQL query to execute.

    Returns:
        list: Results of the query or None if an error occurred.
    """
    conn = None
    try:
        conn = sqlite3.connect(sqlite_file)
        cursor = conn.cursor()
        cursor.execute(query)
        return cursor.fetchall()
    except sqlite3.Error as e:
        logging.error(f"Error executing query {query}: {e}")
        return None
    finally:
        if conn:
            conn.close()


def get_llm_answer_with_context(question: str, sql_result: list, model_name: str) -> list:
    """
    Generate an answer using the LLM, providing the SQL result as context.

    Args:
        question (str): The user's question.
        sql_result (list): The result from the SQL query.
        model_name (str): The name of the LLM model to use.

    Returns:
        list: The LLM-generated answer formatted similarly to the SQL result.
    """
    llm = select_llm(model_name)
    if llm is None:
        logging.error("Invalid LLM model selected.")
        return None

    # Format the SQL result for the LLM
    sql_result_str = format_sql_result(sql_result)

    # Prepare the prompt with formatting instructions
    prompt = f"""
You are provided with the following data extracted from a database query:

{sql_result_str}

Using this data, answer the following question:

{question}

Please ensure that your answer is based solely on the data provided.

**Important Instructions**:

- Format your answer as a **Python list of tuples**, matching the format of the SQL result.
- Do not include any additional text, explanations, or code comments.
- The output should be valid Python syntax that can be parsed using `ast.literal_eval`.
- **Example**:

```python
[('item1',), ('item2',), ...]
"""

    try:
        messages = [
            SystemMessage(content="You are a helpful assistant that answers questions based on provided data."),
            HumanMessage(content=prompt)
        ]
        response = llm.invoke(messages)
        llm_output = response.content.strip()

        # Parse the LLM's output into a list of tuples
        llm_result = parse_llm_output(llm_output)
        if llm_result is None:
            logging.error("LLM output parsing failed.")
        return llm_result
    except Exception as e:
        logging.error(f"Error invoking LLM: {e}")
        return None


def format_sql_result(sql_result: list) -> str:
    """
    Format the SQL result into a readable string for the LLM.

    Args:
        sql_result (list): The result from the SQL query.

    Returns:
        str: Formatted SQL result.
    """
    formatted_result = ''
    for row in sql_result:
        formatted_result += str(row) + '\n'
    return formatted_result.strip()


def parse_llm_output(output_str):
    """
    Parse the LLM's output string into a list of tuples.

    Args:
        output_str (str): The LLM's output string.

    Returns:
        list: Parsed output as a list of tuples.
    """
    try:
        # Preprocess the output to extract the list of tuples
        output_str = output_str.strip()

        # Remove code block markers if present
        if output_str.startswith("```") and output_str.endswith("```"):
            output_str = output_str[3:-3].strip()
            # Remove language specifier if present
            if output_str.lower().startswith("python"):
                output_str = output_str[6:].strip()

        # Extract the list using regex
        match = re.search(r"\[.*\]", output_str, re.DOTALL)
        if match:
            output_str = match.group(0)
        else:
            logging.error("Could not find a list in the LLM output.")
            return None

        # Now attempt to parse
        parsed_output = ast.literal_eval(output_str)
        if isinstance(parsed_output, list):
            # Ensure all elements are tuples
            parsed_output = [tuple(item) if not isinstance(item, tuple) else item for item in parsed_output]
            return parsed_output
        else:
            logging.error("Parsed LLM output is not a list.")
            return None
    except Exception as e:
        logging.error(f"Error parsing LLM output: {e}")
        return None


def calculate_precision_recall(sql_answer, model_answer):
    """
    Calculate precision and recall between SQL and model answers.

    Args:
        sql_answer: The answer from the SQL query.
        model_answer: The answer from the model.

    Returns:
        tuple: Precision and recall values.
    """
    sql_set = set(normalize_response(sql_answer))
    model_set = set(normalize_response(model_answer))

    true_positives = sql_set & model_set
    precision = len(true_positives) / len(model_set) if model_set else 0
    recall = len(true_positives) / len(sql_set) if sql_set else 0

    return precision, recall


def calculate_exact_match(sql_answer, model_answer):
    """
    Calculate exact match and fuzzy match scores between SQL and model answers.

    Args:
        sql_answer: The answer from the SQL query.
        model_answer: The answer from the model.

    Returns:
        tuple: Exact match (1 or 0) and fuzzy match score (0-1).
    """
    sql_answer_str = ' '.join(normalize_response(sql_answer))
    model_answer_str = ' '.join(normalize_response(model_answer))

    exact_match = 1 if sql_answer_str == model_answer_str else 0
    fuzzy_match = fuzz.ratio(sql_answer_str, model_answer_str) / 100.0  # Convert to 0-1 scale
    return exact_match, fuzzy_match


def normalize_response(response):
    """
    Normalize the response data for comparison.

    Args:
        response: The response data to normalize.

    Returns:
        list: Normalized response data as list of strings.
    """
    if isinstance(response, list):
        flat_list = []
        for item in response:
            if isinstance(item, (list, tuple)):
                for sub_item in item:
                    flat_list.append(str(sub_item).lower().strip())
            else:
                flat_list.append(str(item).lower().strip())
        return flat_list
    elif isinstance(response, str):
        return [response.lower().strip()]
    else:
        return [str(response).lower().strip()]
