# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai, os, logging, traceback
import snowflake.connector
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

# Initialize FastAPI app
app = FastAPI()
logging.basicConfig(level=logging.DEBUG)

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

# Allow both tables (UPPERCASE)
ALLOWED_TABLES = {"OHCM_PODCASTS", "AFINA_AD", "CLIENT_MATERIALS"}

def _connect_snowflake():
    return snowflake.connector.connect(
        user=sf_user,
        account=sf_account,
        warehouse=sf_warehouse,
        database=sf_database,
        schema=sf_schema,
        private_key=pkb,
    )

def _sql_for_table(table_name: str, vector_str: str, top_k: int) -> str:
    """Return SQL depending on table structure."""

    if table_name == "AFINA_AD":
        return f"""
        WITH QUERY AS (
            SELECT {vector_str}::VECTOR(FLOAT, 1536) AS QVEC
        )
        SELECT
            FILENAME,
            CHUNK_INDEX,
            CHUNK_TEXT,
            DOI,
            ARTICLE_SUMMARY,
            PDF_SAS_URL,
            TITLE,
            DATE,
            CITATION,
            CITATION_COUNT,
            VECTOR_COSINE_SIMILARITY(EMBEDDING_VECTOR, QVEC) AS SIMILARITY
        FROM {table_name}, QUERY
        ORDER BY SIMILARITY DESC
        LIMIT {top_k};
        """

    # Default structure (all other tables)
    return f"""
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
        VECTOR_COSINE_SIMILARITY(EMBEDDING_VECTOR, QVEC) AS SIMILARITY
    FROM {table_name}, QUERY
    ORDER BY SIMILARITY DESC
    LIMIT {top_k};
    """

def query_snowflake_for_context(query_embedding, table_name, top_k=15):
    # Whitelist check
    table_name = table_name.upper()
    if table_name not in ALLOWED_TABLES:
        raise HTTPException(status_code=403, detail="Table not allowed.")

    # Convert embedding to string for SQL
    vector_str = str(query_embedding)

    # Build SQL (same columns for both tables)
    sql = _sql_for_table(table_name, vector_str, top_k)

    # Execute
    ctx = _connect_snowflake()
    try:
        cur = ctx.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
    finally:
        try:
            cur.close()
        except Exception:
            pass
        ctx.close()

    # Return list of dicts
    return [dict(zip(columns, row)) for row in rows]

# Request model (table name now decided by endpoint, so no table_name here)
class QueryRequest(BaseModel):
    query_text: str
    top_k: int = 15

def _embed(text: str):
    resp = openai.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return resp.data[0].embedding

# --- Two endpoints (GPT will call both separately if needed) ---

@app.post("/sci/query")
async def sci_query(query_data: QueryRequest):
    logging.debug(f"/sci/query received: {query_data}")
    try:
        embedding = _embed(query_data.query_text)
        context = query_snowflake_for_context(embedding, "AFINA_AD", query_data.top_k)
        return {"context": context}
    except Exception as e:
        logging.error(f"/sci/query failed: {e}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Process failed: {str(e)}")

@app.post("/pod/query")
async def pod_query(query_data: QueryRequest):
    logging.debug(f"/pod/query received: {query_data}")
    try:
        embedding = _embed(query_data.query_text)
        context = query_snowflake_for_context(embedding, "OHCM_PODCASTS", query_data.top_k)
        return {"context": context}
    except Exception as e:
        logging.error(f"/pod/query failed: {e}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Process failed: {str(e)}")

@app.post("/client-materials/query")
async def client_materials_query(query_data: QueryRequest):
    logging.debug(f"/client-materials/query received: {query_data}")
    try:
        embedding = _embed(query_data.query_text)
        context = query_snowflake_for_context(embedding,"CLIENT_MATERIALS", query_data.top_k)
        return {"context": context}
    except Exception as e:
        logging.error(f"/client-materials/query failed: {e}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Process failed: {str(e)}")

