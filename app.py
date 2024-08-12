import os
import pandas as pd
import json
import streamlit as st
from pandasai import SmartDataframe
from pandasai.responses.response_parser import ResponseParser
from pandasai.connectors import PostgreSQLConnector, MySQLConnector, SqliteConnector, SQLConnector
from pandasai.ee.connectors import SnowFlakeConnector, DatabricksConnector
from langchain_groq import ChatGroq
from groq import Groq

# Set your API key for Groq
os.environ["GROQ_API_KEY"] = "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"

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
        st.image(result["value"], width=800, use_column_width=True)
        return

    def format_other(self, result):
        st.write(result["value"])
        return

def read_file(file, file_type):
    try:
        if file_type == "csv":
            return read_csv_with_encoding(file)
        elif file_type == "excel":
            return pd.read_excel(file)
        elif file_type == "json":
            data = json.load(file)
            return pd.json_normalize(data)
        else:
            st.error("Unsupported file type.")
            return None
    except Exception as e:
        st.error(f"An error occurred while reading the file: {e}")
        return None

def read_csv_with_encoding(file):
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'utf-16', 'cp1252']
    for encoding in encodings:
        try:
            file.seek(0)
            df = pd.read_csv(file, encoding=encoding)
            if df.empty:
                st.error("The CSV file is empty.")
                return None
            return df
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            st.error("The CSV file is empty or malformed.")
            return None
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return None
    return None

st.title("DATA-CHAD")
st.write("A solution for Data Analysis, No data Analyst Needed")
st.write("# Chat with Dataset")
st.text("This product belongs to NAMA-AI")

# Sidebar for database connection configuration
st.sidebar.title("Database Connection Configuration")
db_type = st.sidebar.selectbox("Select Database Type", ["PostgreSQL", "MySQL", "SQLite", "SQL", "Snowflake", "Databricks"])

if db_type == "PostgreSQL":
    host = st.sidebar.text_input("Host", "localhost")
    port = st.sidebar.number_input("Port", 5432)
    database = st.sidebar.text_input("Database", "mydb")
    username = st.sidebar.text_input("Username", "root")
    password = st.sidebar.text_input("Password", "root", type="password")
    table = st.sidebar.text_input("Table", "payments")
    where = st.sidebar.text_input("Filter (optional)", "[['payment_status', '=', 'PAIDOFF']]")
    print(host+str(port)+database+username+table)
    if st.sidebar.button("Connect PostgreSQL"):
        try:
            postgres_connector = PostgreSQLConnector(
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
            df = SmartDataframe(postgres_connector)
            st.success("Connected to PostgreSQL")
        except Exception as e:
            st.error(f"An error occurred: {e}")

elif db_type == "MySQL":
    host = st.sidebar.text_input("Host", "localhost")
    port = st.sidebar.number_input("Port", 3306)
    database = st.sidebar.text_input("Database", "mydb")
    username = st.sidebar.text_input("Username", "root")
    password = st.sidebar.text_input("Password", "root", type="password")
    table = st.sidebar.text_input("Table", "loans")
    where = st.sidebar.text_input("Filter (optional)", "[['loan_status', '=', 'PAIDOFF']]")
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
        except Exception as e:
            st.error(f"An error occurred: {e}")

elif db_type == "SQLite":
    database = st.sidebar.text_input("Database Path", "path_to_db")
    table = st.sidebar.text_input("Table", "actor")
    where = st.sidebar.text_input("Filter (optional)", "[['first_name', '=', 'PENELOPE']]")
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
        except Exception as e:
            st.error(f"An error occurred: {e}")

# File uploader and chat functionality
uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx", "xls", "json"])

if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1]
    df = read_file(uploaded_file, file_type)

    if df is not None:
        with st.expander("🔎 Dataframe Preview"):
            st.write(df)

        query = st.text_area("🗣️ Chat with Dataframe")
        if query:
            llm = model
            query_engine = SmartDataframe(
                df,
                config={
                    "llm": llm,
                    "response_parser": StreamlitResponse,
                },
            )
            if st.button('Generate'):
                try:
                    answer = query_engine.chat(f"Provide in-depth and accurate analysis of {query}. Try to answer in English.   ")
                    if answer:
                        st.title(f"🤖: {answer}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
else:
    st.write("Please upload a file to start.")
