from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai, os, logging
import snowflake.connector
import traceback
from snowflake import telemetry
from opentelemetry import trace

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

# Initialize FastAPI app
app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")

# Snowflake settings
sf_user = os.getenv("SNOWFLAKE_USER")                
sf_account = os.getenv("SNOWFLAKE_ACCOUNT")          
sf_warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")     
sf_database = os.getenv("SNOWFLAKE_DATABASE")        
sf_schema = os.getenv("SNOWFLAKE_SCHEMA")           

# Load private key and passphrase from environment variables
private_key_str = os.getenv("SNOWFLAKE_PRIVATE_KEY")
private_key_passphrase = os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE")
if private_key_passphrase is not None:
    private_key_passphrase = private_key_passphrase.encode()

p_key = serialization.load_pem_private_key(
    private_key_str.encode(),
    password=private_key_passphrase,
    backend=default_backend()
)

pkb = p_key.private_bytes(
    encoding=serialization.Encoding.DER,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption()
)
logging.basicConfig(level=logging.DEBUG)

# List all allowed tables in UPPERCASE
ALLOWED_TABLES = {"ANAVEX_SCI"}

def query_snowflake_for_context(query_embedding, table_name, top_k=15):
    # Whitelist check for security (ensure uppercase for case-insensitivity)
    table_name = table_name.upper()
    if table_name not in ALLOWED_TABLES:
        raise HTTPException(status_code=403, detail="Table not allowed.")
        
    # Convert embedding to string for SQL
    vector_str = str(query_embedding)

    # Connect to Snowflake using key pair authentication
    ctx = snowflake.connector.connect(
        user=sf_user,
        account=sf_account,
        warehouse=sf_warehouse,
        database=sf_database,
        schema=sf_schema,
        private_key=pkb,   
    )
    
    # Handle different table structures
    if table_name == "TABLE_NAME": # Update table name
        # Snowflake SQL for slide table
        sql = f"""
        WITH QUERY AS (
            SELECT {vector_str}::VECTOR(FLOAT, 1536) AS QVEC
        )

        SELECT
            ID,
        	SOURCE_FILE,
        	TEXT,
        	PAGES,
        	CITATION_COUNT,
        	DOI,
        	TITLE,
        	AUTHORS,
        	PUBLISHED,
        	CITATION,
        	PAGE_REFERENCE,
        	SAS_URL,
        	IS_TABLE,
        	SUMMARY,
            VECTOR_COSINE_SIMILARITY(EMBEDDING_VECTOR, QVEC) AS similarity
        FROM {table_name}, QUERY
        ORDER BY similarity DESC
        LIMIT {top_k};
        """
    
    cursor = ctx.cursor()
    cursor.execute(sql)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description] 
    cursor.close()
    ctx.close()

    # Prepare response as a context list
    context = []
    
    # Format for data
    context = [dict(zip(columns, row)) for row in rows]

    # Return the context as the API response
    return context
    
# Define data model for the request
class QueryRequest(BaseModel):
    query_text: str
    top_k: int = 15
    table_name: str

@app.post("/query/")
async def process_query(query_data: QueryRequest):
    logging.debug(f"Received query data: {query_data}")
    try:
        # Step 1: Generate embedding with OpenAI
        response = openai.embeddings.create(
            input=[query_data.query_text],
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding

        # Step 2: Query Snowflake for top_k similar chunks
        context = query_snowflake_for_context(embedding, query_data.table_name, query_data.top_k)

        return {"context": context}

    except Exception as e:
        logging.error(f"Process failed: {e}")
        logging.error(f"Stack trace: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Process failed: {str(e)}")
