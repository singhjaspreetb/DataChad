import os
import pandas as pd
import streamlit as st
from pandasai import SmartDataframe
from pandasai.responses.response_parser import ResponseParser
from pandasai.connectors import (
    PostgreSQLConnector, MySQLConnector, SqliteConnector, SQLConnector
)
from pandasai.ee.connectors import SnowFlakeConnector,DatabricksConnector
from langchain_groq import ChatGroq
from groq import Groq
import chardet  # For automatic encoding detection

# Set environment variable for Groq API key
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
PANDASAI_API_KEY = os.environ['PANDASAI_API_KEY']

# Initialize Groq client and ChatGroq model
client = Groq()
model = ChatGroq(temperature=0, model_name="llama-3.1-70b-versatile")
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
option = st.sidebar.radio("Select Option", ["File Upload", "Database Connection"])

# Sidebar for toggling between database and file upload


if option == "File Upload":
    st.session_state.show_db = False
    uploaded_file = st.file_uploader("Upload your CSV, JSON, or Excel file", type=["csv", "json", "xlsx", "xls"])

    if uploaded_file:
        # Load and display data
        df = load_file(uploaded_file)

        if df is not None:
            with st.expander("🔎 Dataframe Preview"):
                st.write(df)

            # User input for chat query
            query = st.text_area("🗣️ Chat with Dataframe")
            container = st.container()

            if query:
                try:
                    llm = model
                    query_engine = SmartDataframe(
                        df,
                        config={
                            "llm": llm,
                            "response_parser": StreamlitResponse,
                            "code_to_run":True
                        },
                    )
                    
                    if st.button("GENERATE"):
                        # Process query and display result
                        answer = query_engine.chat(f"provide formatted answer as you are business analyst {query} and provide most suitabole plot, and also explain plots and provide complete plot ignore null and irrelevant values")
                        if answer is not None:
                            st.title(f"👾 Here is your Analysis: {answer}")
                except Exception as e:
                    st.error(f"An error occurred while processing the query: {e}")
    else:
        st.info("Please upload a CSV, JSON, or Excel file to get started.")


elif option == "Database Connection":
    st.session_state.show_db = True
    st.sidebar.title("Database Connection Configuration")
    db_type = st.sidebar.selectbox("Select Database Type", ["PostgreSQL", "MySQL", "SQLite", "SQL", "Snowflake", "Databricks"])

    # Database connection logic goes here (unchanged)

    if db_type == "PostgreSQL":
        host = st.sidebar.text_input("Host", "localhost")  
        port = st.sidebar.number_input("Port", 5432)
        database = st.sidebar.text_input("Database", "database")
        username = st.sidebar.text_input("Username", "user")
        password = st.sidebar.text_input("Password", "password", type="password")
        table = st.sidebar.text_input("Table", "table")
        where = st.sidebar.text_input("Filter (optional)", "")
        query= st.sidebar.text_input("Enter your Query")
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
