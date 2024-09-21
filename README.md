# Enhanced SQL and LLM Hybrid Query App

This project allows you to execute SQL queries, and utilize various language models (LLMs) for enhanced data retrieval and processing. The application was built with reference to the **Galois** paper, and the same queries used in the Galois experiment are included in this project. The query file is located in the `data/Final_Queries.csv` path.

## Prerequisites

Make sure you have [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) and [Python 3.12](https://www.python.org/downloads/release/python-3120/) installed on your machine.

## Installation

Follow these steps to set up the environment and run the application:

### 1. Clone the Repository

To get started, first clone the repository to your local machine:

```bash
git clone https://github.com/Jormart/sql-query-llm.git
```

### 2. Navigate to the Project Directory

Once the repository is cloned, navigate to the project directory:

```bash
cd sql-query-llm
```

### 3. Create virtual environment and install requirements

#### Create a New Environment

To create a new environment for the application, use the following command to install all necessary dependencies from the `environment.yml` file:

```bash
conda env create -f environment.yml
```

#### Activate the Environment

Once the environment is created, activate it:

```bash
conda activate sql-query-llm
```

### 4. Run the Application

Now you can run the application using [Streamlit](https://streamlit.io/). Navigate to the `src` directory and run the following command:

```bash
streamlit run ./src/llm_query_app.py
```

This will start the Streamlit app in your browser. You can now upload the file containing the same queries used in the Galois paper:

1. **Browse** and upload the file: `data/Final_Queries.csv`
2. Select a model and execute queries for enhanced data retrieval and model comparison.

## Results Analysis

For detailed results analysis and model comparison, please refer to the Jupyter notebook `model_comparison.ipynb` located in the `data/results` directory. This notebook contains code and visualizations to evaluate the performance of different models based on the queries executed.

## Features

- **Hybrid SQL and LLM Queries**: Combine structured SQL queries with unstructured data retrieved from language models.
- **Multiple LLMs Supported**: Choose between different language models such as OpenAI's GPT and Groq's models.
- **Customizable Query Execution**: Upload custom queries and databases, and visualize results with precision and recall metrics.
- **Galois Paper Queries**: The app allows you to run the same SQL queries used in the Galois research paper to compare results.

## License

This project is licensed under the MIT License.
