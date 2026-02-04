import os
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

model = SentenceTransformer('all-MiniLM-L6-v2')
URL_BANCO = os.getenv("DATABASE_URL")

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
    #implemented RAG in the Agen
    cursor.close()
    connection.close()

    return result[0] if result else "I didn't find any FAQ information."