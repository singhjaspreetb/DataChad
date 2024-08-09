import os
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
# from pandasai.callbacks import BaseCallback
from pandasai.responses.response_parser import ResponseParser
from groq import Groq
from langchain_groq import ChatGroq
os.environ["GROQ_API_KEY"] = "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"

client = Groq()
model = ChatGroq(temperature=0, model_name="llama-3.1-70b-versatile")
# class StreamlitCallback(BaseCallback):
#     def __init__(self, container) -> None:
#         """Initialize callback handler."""
#         self.container = container

#     def on_code(self, response: str):
#         self.container.code(response)

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
        except UnicodeDecodeError as e:
            st.error(f"UnicodeDecodeError with encoding {encoding}: {e}")
        except pd.errors.EmptyDataError:
            st.error("The CSV file is empty or malformed.")
            return None
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return None
    return None



st.title("DATA-CHAD")
st.write("A solution for Data Analysis, No data Analyst Neede")
st.write("# Chat with Dataset")
st.text("This product belongs to NAMA-AI")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    df = read_csv_with_encoding(uploaded_file)
    
    with st.expander("🔎 Dataframe Preview"):
        st.write(df)

    query = st.text_area("🗣️ Chat with Dataframe")
    container = st.container()

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
          answer = query_engine.chat(query)
          if answer!=None:
            st.title(f"🤖: {answer}")
else:
    st.write("Please upload a CSV file to start.")
