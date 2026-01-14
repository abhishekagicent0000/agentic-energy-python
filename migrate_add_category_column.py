"""
Simple migration to add `category` column to `operation_suggestion` PostgreSQL table.

Run this script on the machine that has access to the PostgreSQL `DATABASE_URL` configured
in the project's environment (it uses the same `get_pg_connection()` behavior as `operation.py`).
"""
import os
import logging
from dotenv import load_dotenv
import psycopg2

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv('DATABASE_URL')


def get_conn():
    if not DATABASE_URL:
        logger.error('DATABASE_URL not set in environment')
        return None
    from urllib.parse import urlparse, urlunparse
    parsed = urlparse(DATABASE_URL)
    clean_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, '', parsed.fragment))
    try:
        return psycopg2.connect(clean_url)
    except Exception as e:
        logger.error(f'Failed to connect: {e}')
        return None


def add_category_column():
    conn = get_conn()
    if not conn:
        return 1
    cur = conn.cursor()
    try:
        # safe-add: check information_schema first
        cur.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'operation_suggestion' AND column_name = 'category'
        """)
        if cur.fetchone():
            logger.info('column `category` already exists on operation_suggestion')
            return 0

        # add column with default
        cur.execute("ALTER TABLE operation_suggestion ADD COLUMN category VARCHAR(50) DEFAULT 'Production';")
        conn.commit()
        logger.info('Added `category` column to operation_suggestion')
        return 0
    except Exception as e:
        logger.exception('Failed to add category column')
        conn.rollback()
        return 2
    finally:
        cur.close()
        conn.close()


if __name__ == '__main__':
    exit(add_category_column())
