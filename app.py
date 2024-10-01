import os
import pandas as pd
import streamlit as st
import ast
import pymongo
import json
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from pandasai import SmartDataframe
from pandasai.responses.response_parser import ResponseParser
from pandasai.connectors import (
    PostgreSQLConnector, MySQLConnector, SqliteConnector, SQLConnector
)
from pandasai.ee.connectors import SnowFlakeConnector, DatabricksConnector
from langchain_groq import ChatGroq
from groq import Groq
import chardet  # For automatic encoding detection


GROQ_API_KEY = os.environ["GROQ_API_KEY"]
PANDASAI_API_KEY = os.environ['PANDASAI_API_KEY']

llm = Groq()
llm_types = [
    "llama-3.2-90b-text-preview",
    "llama-3.2-11b-text-preview",
    "llama-3.2-3b-preview",
    "llama-3.2-1b-preview",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-groq-70b-8192-tool-use-preview",
    "llama3-groq-8b-8192-tool-use-preview",
    "llama-guard-3-8b",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma-7b-it",
    "gemma2-9b-it"
]
llm_type = st.sidebar.selectbox("Select Model", llm_types)

model = ChatGroq(temperature=0, model_name=llm_type)


def prompt_user(user_input):
    return f"""# Data Analytics Expert Prompt

        You are an expert data analyst with extensive experience in business intelligence, statistical analysis, and data-driven decision making. Your role is to assist users in extracting meaningful insights from their data to solve complex business problems and drive strategic decisions.

        ## Your Expertise:

        1. Data Analysis Techniques:
        - Descriptive, diagnostic, predictive, and prescriptive analytics
        - Statistical analysis (regression, hypothesis testing, time series analysis)
        - Machine learning algorithms (classification, clustering, anomaly detection)
        - Data mining and pattern recognition

        2. Business Intelligence:
        - KPI development and tracking
        - Dashboard creation and data visualization
        - Performance measurement and benchmarking
        - Competitive analysis and market trends

        3. Data Management:
        - Data quality assessment and cleansing
        - ETL processes and data integration
        - Database management and SQL querying
        - Big data technologies (Hadoop, Spark)

        4. Industry-Specific Knowledge:
        - Retail, finance, healthcare, manufacturing, and technology sectors
        - Industry-specific metrics and benchmarks
        - Regulatory compliance and data governance

        ## Your Approach:

        1. Problem Definition:
        - Clearly define the business problem or question
        - Identify key stakeholders and their requirements
        - Determine the scope and limitations of the analysis

        2. Data Assessment:
        - Evaluate available data sources and their relevance
        - Assess data quality, completeness, and reliability
        - Identify any data gaps or additional data needs

        3. Analysis Planning:
        - Determine appropriate analytical techniques
        - Outline the step-by-step analysis process
        - Consider potential challenges and mitigation strategies

        4. Data Exploration and Preprocessing:
        - Perform exploratory data analysis (EDA)
        - Clean and preprocess data as necessary
        - Create derived variables or features if needed

        5. In-depth Analysis:
        - Apply chosen analytical techniques
        - Validate results and test for statistical significance
        - Iterate and refine the analysis as needed

        6. Interpretation and Insights:
        - Translate analytical findings into business insights
        - Identify actionable recommendations
        - Quantify potential impact and ROI of recommendations

        7. Communication and Visualization:
        - Create clear, concise visualizations to support findings
        - Develop a narrative that tells the data story
        - Tailor communication to the audience's technical level

        8. Implementation and Monitoring:
        - Propose an action plan for implementing recommendations
        - Suggest metrics for monitoring the impact of changes
        - Recommend follow-up analyses or data collection

        ## Your Output:

        When responding to user queries, provide:

        1. A summary of the problem and approach
        2. Key findings and insights, supported by data
        3. Actionable recommendations with potential impact
        4. Visualizations or tables to illustrate important points
        5. Limitations of the analysis and areas for further investigation
        6. Next steps or follow-up questions to deepen the analysis

        Remember to always consider the business context, potential biases, and ethical implications of your analysis. Strive to provide accurate, unbiased, and actionable insights that drive measurable business value.

        ## User Query:

        {user_input}

        Please address the above user query using your expertise and approach outlined in this prompt. Provide a comprehensive analysis and actionable recommendations based on the given information."""
class StreamlitResponse(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        st.dataframe(result["value"])
        return

    def format_plot(self, result):
        st.image(result["value"], width=900, use_column_width=True)
        return

    def format_other(self, result):
        st.write(result["value"])
        return

def detect_encoding(file):
    # Detect file encoding
    raw_data = file.read(1024)  # Read a small portion of the file
    result = chardet.detect(raw_data)
    file.seek(0)  # Reset file pointer to the beginning
    return result['encoding']

def read_csv_with_encoding(file):
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'utf-16', 'cp1252']
    for encoding in encodings:
        try:
            file.seek(0)
            df = pd.read_csv(file, encoding=encoding)
            if not df.empty:
                return df
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            st.error("The CSV file is empty or malformed.")
            return None
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return None
    st.error("Unable to decode the CSV file with common encodings.")
    return None

def load_file(file) -> pd.DataFrame:
    if file.type == "text/csv":
        df = read_csv_with_encoding(file)
    elif file.type == "application/json":
        try:
            df = pd.read_json(file)
        except ValueError as e:
            st.error(f"An error occurred while reading JSON file: {e}")
            return None
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return None
    elif file.type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
        try:
            df = pd.read_excel(file, engine='openpyxl')  # Use engine for xlsx files
        except Exception as e:
            st.error(f"An error occurred while reading Excel file: {e}")
            return None
    else:
        st.error("Unsupported file type. Please upload a CSV, JSON, or Excel file.")
        return None

    return df

def get_document_structure(doc):
    """
    Converts the structure of a MongoDB document into a text representation.
    
    Args:
    - doc (dict): The MongoDB document.
    
    Returns:
    - str: A text representation of the document structure.
    
    """
    lines = []
    for key, value in doc.items():
        # Get the type of the value (e.g., str, int, list)
        value_type = type(value).__name__
        lines.append(f"{key}: {value_type}")
    
    # Join lines into a single string
    structure_text = "\n".join(lines)
    return structure_text

def preprocess_json_string(json_string):
    # Replace single quotes with double quotes
    json_string = json_string.replace("'", '"')
    # Ensure keys are enclosed in double quotes
    # This is a simple fix; more complex adjustments might be needed based on your output
    if json_string.startswith("{") and json_string.endswith("}"):
        return json_string
    return f"{{{json_string}}}"

def mon_query(ques,structure,database,collection):
    prompt=f"""
    Task: You are a Mongodb query expert, you are given with the document structure.
    Databse is already connected.
    You primary task is to understand users task, based on the given information provide
    the correct query to fetch the required data.

    structure:{structure}
    query:{ques}
    database:{database}
    collection:{collection}

    response:
    Only provide query to extract data.

    collection.find_one({{"name": "alice"}})

    Requirements:
    Only provide the query no explainantion
    Only provide mongodb query.
    follow structure of document
    Use collection to get the query
    never provide explaination
    only mongodb query.
    """

    completion = llm.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model=llm_type,
    )

    return completion.choices[0].message.content

# Initialize session state variables
if "show_db" not in st.session_state:
    st.session_state.show_db = False
if "df" not in st.session_state:
    st.session_state.df = None

# Streamlit application
st.write("# DATA CHAD")
st.write("This product belongs to NAMA AI")

# Sidebar for toggling between database and file upload
st.sidebar.title("Navigation")
option = st.sidebar.radio("Select Option", ["File Upload", "Database Connection"], key="option_radio")

if option == "File Upload":
    st.session_state.show_db = False
    uploaded_file = st.file_uploader("Upload your CSV, JSON, or Excel file", type=["csv", "json", "xlsx", "xls"])

    if uploaded_file:
        df = load_file(uploaded_file)
        st.session_state.df = df

        if df is not None:
            with st.expander("🔎 Dataframe Preview"):
                st.write(df)
            query = st.text_area("🗣️ Chat with Dataframe")
            if query:
                try:
                    llm = model
                    query_engine = SmartDataframe(
                        df,
                        config={
                            "llm": llm,
                            "response_parser": StreamlitResponse,
                            "code_to_run": True
                        },
                    )
                    if st.button("GENERATE"):
                        answer = query_engine.chat(f"Provide formatted answer as you are a business analyst: {query} and provide the most suitable plot. Also, explain plots and provide a complete plot, ignoring null and irrelevant values.")
                        if answer is not None:
                            st.title(f"👾 Here is your Analysis: {answer}")
                except Exception as e:
                    st.error(f"An error occurred while processing the query: {e}")
    else:
        st.info("Please upload a CSV, JSON, or Excel file to get started.")

elif option == "Database Connection":
    st.session_state.show_db = True
    st.sidebar.title("Database Connection Configuration")
    db_type = st.sidebar.selectbox("Select Database Type", ["PostgreSQL", "MySQL", "SQLite", "SQL", "Snowflake", "Databricks","MongoDB"])

    # Database connection logic goes here (unchanged)

    if db_type == "PostgreSQL":
        host = st.sidebar.text_input("Host", "postgres")  
        port = st.sidebar.number_input("Port", 5432)
        database = st.sidebar.text_input("Database", "mydb")
        username = st.sidebar.text_input("Username", "root")
        password = st.sidebar.text_input("Password", "root", type="password")
        table = st.sidebar.text_input("Table", "payments")
        where = st.sidebar.text_input("Filter (optional)", "")
        query= st.sidebar.text_input("Enter your Query")
        query = prompt_user(query)
        if st.sidebar.button("Connect PostgreSQL"):
            try:
                where_clause = eval(where) if where else None
                postgres_connector = PostgreSQLConnector(
                    config={
                        "host": host,
                        "port": port,
                        "database": database,
                        "username": username,
                        "password": password,
                        "table": table,     
                        "where": where_clause,
                    }
                )
                df = SmartDataframe(postgres_connector)
                st.session_state.df = df
                st.success("Connected to PostgreSQL")
                answer = df.chat(query)
                st.title(answer)
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

    elif db_type == "MySQL":
        host = st.sidebar.text_input("Host", "localhost")
        port = st.sidebar.number_input("Port", 3306)
        database = st.sidebar.text_input("Database", "mydb")
        username = st.sidebar.text_input("Username", "root")
        password = st.sidebar.text_input("Password", "root", type="password")
        table = st.sidebar.text_input("Table", "loans")
        where = st.sidebar.text_input("Filter (optional)", "[['loan_status', '=', 'PAIDOFF']]")
        query= st.sidebar.text_input("Enter your Query")
        query = prompt_user(query)
        if st.sidebar.button("Connect MySQL"):
            try:
                mysql_connector = MySQLConnector(
                    config={
                        "host": host,
                        "port": port,
                        "database": database,
                        "username": username,
                        "password": password,
                        "table": table,
                        "where": eval(where),
                    }
                )
                df = SmartDataframe(mysql_connector)
                st.success("Connected to MySQL")
                answer = df.chat(query)
                st.title(answer)
            except Exception as e   :
                st.error(f"An error occurred: {e}")

    elif db_type == "SQLite":
        database = st.sidebar.text_input("Database Path", "path_to_db")
        table = st.sidebar.text_input("Table", "actor")
        where = st.sidebar.text_input("Filter (optional)", "[['first_name', '=', 'PENELOPE']]")
        query= st.sidebar.text_input("Enter your Query")
        query = prompt_user(query)
        if st.sidebar.button("Connect SQLite"):
            try:
                sqlite_connector = SqliteConnector(
                    config={
                        "database": database,
                        "table": table,
                        "where": eval(where),
                    }
                )
                df = SmartDataframe(sqlite_connector)
                st.success("Connected to SQLite")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    elif db_type == "SQL":
        dialect = st.sidebar.text_input("Dialect", "sqlite")
        driver = st.sidebar.text_input("Driver", "pysqlite")
        host = st.sidebar.text_input("Host", "localhost")
        port = st.sidebar.number_input("Port", 3306)
        database = st.sidebar.text_input("Database", "mydb")
        username = st.sidebar.text_input("Username", "root")
        password = st.sidebar.text_input("Password", "root", type="password")
        table = st.sidebar.text_input("Table", "loans")
        where = st.sidebar.text_input("Filter (optional)", "[['loan_status', '=', 'PAIDOFF']]")
        query= st.sidebar.text_input("Enter your Query")
        query = prompt_user(query)
        if st.sidebar.button("Connect SQL"):
            try:
                sql_connector = SQLConnector(
                    config={
                        "dialect": dialect,
                        "driver": driver,
                        "host": host,
                        "port": port,
                        "database": database,
                        "username": username,
                        "password": password,
                        "table": table,
                        "where": eval(where),
                    }
                )
                df = SmartDataframe(sql_connector)
                st.success("Connected to SQL")
                answer = df.chat(query)
                st.title(answer)
            except Exception as e:
                st.error(f"An error occurred: {e}")

    elif db_type == "Snowflake":
        account = st.sidebar.text_input("Account", "your_account")
        database = st.sidebar.text_input("Database", "SNOWFLAKE_SAMPLE_DATA")
        username = st.sidebar.text_input("Username", "test")
        password = st.sidebar.text_input("Password", "*****", type="password")
        table = st.sidebar.text_input("Table", "lineitem")
        warehouse = st.sidebar.text_input("Warehouse", "COMPUTE_WH")
        dbSchema = st.sidebar.text_input("Schema", "tpch_sf1")
        where = st.sidebar.text_input("Filter (optional)", "[['l_quantity', '>', '49']]")
        query= st.sidebar.text_input("Enter your Query")
        query = prompt_user(query)
        if st.sidebar.button("Connect Snowflake"):
            try:
                snowflake_connector = SnowFlakeConnector(
                    config={
                        "account": account,
                        "database": database,
                        "username": username,
                        "password": password,
                        "table": table,
                        "warehouse": warehouse,
                        "dbSchema": dbSchema,
                        "where": eval(where),
                    }
                )
                df = SmartDataframe(snowflake_connector)
                st.success("Connected to Snowflake")
                answer = df.chat(query)
                st.title(answer)
            except Exception as e:
                st.error(f"An error occurred: {e}")

    elif db_type == "Databricks":
        host = st.sidebar.text_input("Host", "adb-*****.azuredatabricks.net")
        database = st.sidebar.text_input("Database", "default")
        token = st.sidebar.text_input("Token", "dapidfd412321", type="password")
        port = st.sidebar.number_input("Port", 443)
        table = st.sidebar.text_input("Table", "loan_payments_data")
        httpPath = st.sidebar.text_input("HTTP Path", "/sql/1.0/warehouses/213421312")
        where = st.sidebar.text_input("Filter (optional)", "[['loan_status', '=', 'PAIDOFF']]")
        query= st.sidebar.text_input("Enter your Query")
        query = prompt_user(query)
        if st.sidebar.button("Connect Databricks"):
            try:
                databricks_connector = DatabricksConnector(
                    config={
                        "host": host,
                        "database": database,
                        "token": token,
                        "port": port,
                        "table": table,
                        "httpPath": httpPath,
                        "where": eval(where),
                    }
                )
                df = SmartDataframe(databricks_connector)
                st.success("Connected to Databricks")
                answer = df.chat(query)
                st.title(answer)
            except Exception as e:
                st.error(f"An error occurred: {e}")

    elif db_type == "MongoDB":
        host = st.sidebar.text_input("Host", "localhost:27017")
        username = st.sidebar.text_input("Username", value="", key="mongo_user")
        password = st.sidebar.text_input("Password", value="", type="password", key="mongo_pass")
        database = st.sidebar.text_input("Database Name")
        collect = st.sidebar.text_input("Collection Name")
        query = st.sidebar.text_input("Enter your Query")
        if st.sidebar.button("Connect MongoDB"):
            try:
                # Construct the MongoDB connection string with authentication
                mongo_uri = f"mongodb://{username}:{password}@{host}"
                client = MongoClient(mongo_uri)

                # Access the specified database and collection
                db = client[database]
                collection = db[collect]

                # Test the connection
                client.server_info()
                st.success("Connected to MongoDB")

                ans=(f"Collection 'my_new_collection' created with document: {collection.find_one()}")
                document_structure = collection.find_one()
                if document_structure:
                    structure = get_document_structure(document_structure)
                    generated_query = mon_query(query, structure,database,collection)
                    ans = eval(generated_query)
                    document = ans
                    st.write("Found document:", document)
            except ConnectionFailure as e:
                st.error(f"An error occurred: {e}")



    # Similar logic for other databases...

if st.session_state.df is not None:
    with st.container():
        query = st.text_area("🗣️ Chat with Dataframe", key="chat_query")
        if query:
            try:
                llm = model
                query_engine = SmartDataframe(
                    st.session_state.df,
                    config={
                        "llm": llm,
                        "response_parser": StreamlitResponse,
                        "code_to_run": True,
                        "use_vector_store": False
                    },
                )
                if st.button("GENERATE", key="generate_button"):
                    try:
                        answer = query_engine.chat(f"Provide formatted answer as you are a business analyst: {query} and provide the most suitable plot. Also, explain plots and provide a complete plot, ignoring null and irrelevant values.")
                        if answer is not None:
                            st.title(f"👾 Here is your Analysis: {answer}")
                    except pandasai.exceptions.PandasAIApiCallError as e:
                        st.error(f"PandasAI API error: {str(e)}. Please try again later.")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {str(e)}")
            except Exception as e:
                st.error(f"An error occurred while processing the query: {e}")