import os
import openai
import streamlit as st
import pandas as pd
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI, OpenAI
from dotenv import load_dotenv

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
print(os.environ.get('OPENAI_API_KEY')) 
openai.api_key = os.environ['OPENAI_API_KEY']

def preprocess_csv(file):
    df = pd.read_csv(file)
    return df

# Define a function to chunk the data
def chunk_data(df, chunk_size=500):
    chunks = []
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunk = df[i*chunk_size:(i+1)*chunk_size]
        chunks.append(chunk)
    return chunks

def main():
    st.set_page_config(page_title="Ask your CSV")
    st.header("Ask your CSV")

    user_csv = st.file_uploader("Upload your CSV file", type="csv")

    if user_csv is not None:
        user_question = st.text_input("Ask a question about your CSV:")

         # Preprocess the CSV file and chunk the data
        df = preprocess_csv(user_csv)
        chunks = chunk_data(df)

        import openai
        openai.api_key = os.getenv('OPENAI_API_KEY')

        llm = OpenAI(temperature=0)
        if user_question:
            response = handle_question(chunks, user_question, llm)
            st.write(response)

def handle_question(chunks, question, llm):
    responses = []
    for chunk in chunks:
        data_str = chunk.to_string()
        input_text = f"Data:\n{data_str}\nQuestion: {question}"
        
        # Initialize LangChain agent with the chunk data
        agent = create_csv_agent(llm, chunk, verbose=True, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, allow_dangerous_code=True)
        
        response = agent.run(question)
        responses.append(response)
        
    return "\n".join(responses)



if __name__ == "__main__":
    main()
