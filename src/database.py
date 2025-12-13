import json
import logging
import psycopg2
import os
from dotenv import load_dotenv

logger = logging.getLogger("detection")
logger.setLevel(logging.INFO)

# Load environment variables
load_dotenv()

# PostgreSQL connection
try:
    conn = psycopg2.connect(
        host=os.environ["DB_HOST"],
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        port=int(os.environ.get("DB_PORT", 5432))
    )
    conn.autocommit = True
    cursor = conn.cursor()
    logger.info("✅ Connected to PostgreSQL")
except Exception as e:
    logger.error(f"❌ Failed to connect to PostgreSQL: {e}")
    raise


def insert_model_details(data: dict):
    """
    Insert model details into model_details table.
    
    Expected fields in `data`:
    - org_id
    - user_id
    - name
    - model_weight_path
    """

    try:
        with conn.cursor() as cursor:
            insert_query = """
                INSERT INTO model_details (
                    org_id, user_id, name, model_weight_path
                )
                VALUES (%s, %s, %s, %s)
                RETURNING id;
            """

            cursor.execute(
                insert_query,
                (
                    data["org_id"],
                    data["user_id"],
                    data["name"],
                    data["model_weight_path"]
                )
            )

            inserted_id = cursor.fetchone()[0]
            conn.commit()
            logger.info(f"✅ Model  details stored with id={inserted_id}")
            return inserted_id

    except Exception as e:
        conn.rollback()
        logger.error(f"❌ Failed to insert model details: {e}")
        return None




def get_models_by_org_user(org_id: int, user_id: int):
    """
    Fetch all model details for a given org_id and user_id.
    """
    conn = None
    cursor = None

    try:
        # Create a new DB connection each time
        conn = psycopg2.connect(
        host=os.environ["DB_HOST"],
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        port=int(os.environ.get("DB_PORT", 5432))
      )
        cursor = conn.cursor()

        query = """
            SELECT id, name, model_weight_path, created_at
            FROM model_details
            WHERE org_id = %s AND user_id = %s
        """
        
        cursor.execute(query, (org_id, user_id))
        rows = cursor.fetchall()

        if not rows:
            logger.warning(f"No models found for org_id={org_id} and user_id={user_id}")
            return []

        models = [
            {
                "id": row[0],
                "name": row[1],
                "model_weight_path": row[2],
                "created_at": row[3].isoformat() if row[3] else None
            }
            for row in rows
        ]

        return models

    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        raise

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    
