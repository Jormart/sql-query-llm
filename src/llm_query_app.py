# llm_query_app.py

import os
import logging

import pandas as pd
from dotenv import load_dotenv
import streamlit as st

from llm_utils import (
    get_llm_answer_direct,
    get_llm_answer_with_context,
    execute_query,
    calculate_precision_recall,
    calculate_exact_match,
)
from sql_translator import translate_to_human

# Load environment variables from the .env file
load_dotenv()

# Setup logging to capture events and useful information
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


def read_queries_from_csv(csv_file: str) -> pd.DataFrame:
    """
    Load queries from a CSV file into a DataFrame.

    Args:
        csv_file (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the queries with columns 'Question', 'Query', and 'Database'.
    """
    df = pd.read_csv(csv_file)
    return df[['Question', 'Query', 'Database']]


def get_response(
    sqlite_file: str,
    sql_query: str,
    question: str,
    model_name: str
):
    """
    Get responses by executing the SQL query, translating it, and generating answers with the LLM.

    Args:
        sqlite_file (str): Path to the SQLite database file.
        sql_query (str): The SQL query to execute.
        question (str): The original user question.
        model_name (str): The name of the LLM model to use.

    Returns:
        tuple: Contains query_answer, translated_query, llm_answer, single_question_answer.
    """
    # Execute the SQL query on the specified database
    query_answer = execute_query(sqlite_file, sql_query)
    if query_answer is None:
        return None, None, None, None

    # Translate the SQL query into natural language
    translated_query = translate_to_human(sql_query, model_name)

    # Use the translated_query as input to the LLM to generate a contextual answer
    llm_answer = get_llm_answer_with_context(translated_query, query_answer, model_name)

    # Get a direct answer to the original question using the LLM
    single_question_answer = get_llm_answer_direct(question, model_name)

    return query_answer, translated_query, llm_answer, single_question_answer


def main():
    """
    Main function to run the Streamlit app.
    """
    # Configure the Streamlit page with title and icon
    st.set_page_config(page_title="Enhanced SQL and LLM Hybrid Query App", page_icon=":bar_chart:")

    st.title("Enhanced SQL and LLM Hybrid Query App")

    # Allow the user to upload a CSV file containing queries
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is None:
        st.warning("Please upload a CSV file to proceed.")
        return

    # Read queries from the uploaded CSV file
    queries_df = read_queries_from_csv(uploaded_file)
    st.dataframe(queries_df[['Query', 'Database', 'Question']])

    # Allow the user to select the LLM model from the sidebar
    model_name = st.sidebar.selectbox(
        "Choose a model",
        ["gpt-3.5-turbo", "gpt-4o-mini", "llama-3.1-70b-versatile", "llama-3.1-8b-instant"]
    )

    # Define the paths to the available SQLite databases
    db_files = {
        'flight_2': 'data/db/flight_2.sqlite',
        'flight_4': 'data/db/flight_4.sqlite',
        'geo': 'data/db/geo.sqlite',
        'singer': 'data/db/singer.sqlite',
        'world_1': 'data/db/world_1.sqlite'
    }

    # Initialize a list to store the results
    results = []
    progress_bar = st.progress(0)
    total_queries = len(queries_df)

    # Button to execute the queries
    if st.button("Execute Queries"):
        for index, row in queries_df.iterrows():
            selected_db = row['Database']
            if selected_db in db_files:
                sqlite_file = db_files[selected_db]
                sql_query = row['Query']
                question = row['Question']  # Get the original question

                # Get the responses by executing the query and using the LLM
                query_answer, translated_query, llm_answer, single_question_answer = get_response(
                    sqlite_file, sql_query, question, model_name
                )

                # Verify that all responses were obtained successfully
                if query_answer is not None and llm_answer is not None and single_question_answer is not None:
                    # Calculate precision, recall, exact match, and fuzzy match metrics
                    precision, recall = calculate_precision_recall(query_answer, llm_answer)
                    exact_match, fuzzy_match = calculate_exact_match(query_answer, llm_answer)

                    # Append the results to the list
                    results.append({
                        "Query": sql_query,
                        "Database": selected_db,
                        "Query Answer": str(query_answer),
                        "Translated Query (Natural Language)": translated_query,
                        "LLM Answer": str(llm_answer),
                        "Question": question,
                        "Single Question Answer": single_question_answer,  # Added new column
                        "Precision": precision,
                        "Recall": recall,
                        "Exact Match": exact_match,
                        "Fuzzy Match": fuzzy_match
                    })

                    # Update the progress bar
                    progress_bar.progress(
                        (index + 1) / total_queries,
                        text=f"Processing query {index + 1} of {total_queries}"
                    )
                    logging.info(f"Query {index + 1} processed successfully")
                else:
                    st.error(f"Error processing query {index + 1}")
            else:
                logging.error(f"Database {selected_db} not found in db_files.")
                st.error(f"Database {selected_db} not found.")

        # If there are results, display them and allow the user to download them
        if results:
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)

            # Define the storage path in 'data/results'
            results_folder = 'data/results'
            os.makedirs(results_folder, exist_ok=True)  # Create folder if it doesn't exist
            results_file_path = os.path.join(results_folder, f'{model_name}_results.csv')

            # Save the CSV file in 'data/results'
            results_df.to_csv(results_file_path, index=False)

            # Display a message with the path where the file was saved
            st.success(f"Results saved in {results_file_path}")

            # Allow the user to download the CSV file
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"Download {model_name} Results",
                data=csv,
                file_name=f'{model_name}_results.csv'
            )


if __name__ == "__main__":
    main()
