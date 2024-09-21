# Enhanced SQL and LLM Hybrid Query App

This project enables the execution of SQL queries and allows for enhanced language model (LLM) data retrieval, leveraging multiple models for hybrid query processing.

### Background
The approach is inspired by the Galois research project, which introduces a way to combine traditional SQL-based databases with large language models to expand the scope of data retrieval. This application builds on these principles to offer a user-friendly interface for hybrid queries, allowing users to ask more complex, natural language queries that are then translated into SQL.

You can find more details on the Galois project and repository here: [Galois Repository](https://gitlab.eurecom.fr/saeedm1/galois).

---

## Prerequisites

Ensure that you have [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) and [Python 3.12](https://www.python.org/downloads/release/python-3120/) installed on your system.

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Jormart/sql-query-llm.git
```

### Step 2: Navigate to the Project Directory

```bash
cd sql-query-llm
```

### Step 3: Create Virtual Environment and Install Requirements

#### Create a New Environment

To set up the environment and install the necessary packages:

```bash
conda env create -f environment.yml
```

#### Activate the Environment

```bash
conda activate sql-query-llm
```

---

## Running the Application

1. Navigate to the `src` directory:

   ```bash
   cd src
   ```

2. Run the Streamlit application:

   ```bash
   streamlit run llm_query_app.py
   ```

The app will open in your browser, allowing you to upload CSV files, select models, and execute SQL queries. The results will be displayed along with several evaluation metrics (Precision, Recall, Exact Match, Fuzzy Match).

---

## Features

- **Hybrid Query Processing**: Combines SQL queries with LLMs to expand the range of possible queries.
- **Multiple LLM Support**: Choose from various language models such as GPT and Llama models.
- **CSV File Input**: Easily upload a CSV file containing your queries and databases.
- **Performance Metrics**: Evaluate the precision, recall, and exact match of query results.
- **Custom Query Execution**: Execute both SQL-based queries and LLM responses, with a detailed comparison of results.

---

## Results and Analysis

Results from your query executions can be found in the `data/results` folder. Each model's results are stored in separate CSV files.

For detailed analysis and comparisons, use the provided Jupyter notebook `model_comparison.ipynb` located in the `data/results` folder. The notebook allows for a thorough comparison of the models, including metrics such as precision, recall, and exact match against Galois.

---

## License

This project is licensed under the MIT License.
