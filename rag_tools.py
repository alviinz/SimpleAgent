import os
from langchain_openai import OpenAI
from langchain_groq import ChatGroq
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

model = SentenceTransformer('all-MiniLM-L6-v2')
URL_BANCO = os.getenv("DATABASE_URL")

llm_verifier = ChatGroq(
    model='llama-3.3-70b-versatile',
    api_key=os.getenv("GROQ_API_KEY"),
)

def search_faq_database(user_question):
    connection = psycopg2.connect(URL_BANCO)
    register_vector(connection)
    cursor = connection.cursor()

    vector_question = model.encode(user_question).tolist()

    # busca por similaridade
    # operador <-> calcula a distância entre os vetores salvos e a pergunta
    comand = "SELECT answer FROM faq_knowledge ORDER BY vector <-> %s::vector LIMIT 1"
    
    cursor.execute(comand, (vector_question,))
    result = cursor.fetchone()
    cursor.close()
    connection.close()

    return result[0] if result else "I didn't find any FAQ information."

def verify_response_quality(user_question, context):
    """Verifier (Response Verifier)"""
    if not context or "Database error" in context:
        return False
        
    prompt = (
        f"Question: {user_question}\n"
        f"Retrieved Context: {context}\n"
        f"Does the above context contain the direct answer to the question? Respond ONLY 'YES' or 'NO'."
    )
    
    decision = llm_verifier.invoke([HumanMessage(content=prompt)]).content.upper()
    return "YES" in decision