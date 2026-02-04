import os
import json
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

model = SentenceTransformer('all-MiniLM-L6-v2')

URL_BD = os.getenv("DATABASE_URL")
connection = psycopg2.connect(URL_BD)
connection.autocommit = True
cursor = connection.cursor()

cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
register_vector(connection)
cursor.execute("DROP TABLE IF EXISTS faq_knowledge")
cursor.execute("CREATE TABLE faq_knowledge (question text, answer text, vector vector(384))")

with open('faq.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("Loading data into the database...")
for item in data:
    question = item['user_query']
    answer = item['answer_default']
    
    vector = model.encode(question).tolist()
    
    cursor.execute(
        "INSERT INTO faq_knowledge (question, answer, vector) VALUES (%s, %s, %s)",
        (question, answer, vector)
    )

cursor.close()
connection.close()
print("Success. Database is ready now.")