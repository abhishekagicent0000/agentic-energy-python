import os
from snowflake.connector import connect as snowflake_connect
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Snowflake Configuration
SNOWFLAKE_CONFIG = {
    'user': os.getenv('SNOWFLAKE_USER'),
    'password': os.getenv('SNOWFLAKE_PASSWORD'),
    'account': os.getenv('SNOWFLAKE_ACCOUNT'),
    'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE'),
    'database': os.getenv('SNOWFLAKE_DATABASE'),
    'schema': os.getenv('SNOWFLAKE_SCHEMA'),
    'role': os.getenv('SNOWFLAKE_ROLE')
}

def init_db():
    try:
        conn = snowflake_connect(**SNOWFLAKE_CONFIG)
        cursor = conn.cursor()
        
        # Create database if it doesn't exist
        print(f"Creating database {SNOWFLAKE_CONFIG['database']}...")
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {SNOWFLAKE_CONFIG['database']}")
        
        # Create schema if it doesn't exist
        print(f"Creating schema {SNOWFLAKE_CONFIG['schema']}...")
        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {SNOWFLAKE_CONFIG['database']}.{SNOWFLAKE_CONFIG['schema']}")
        
        # Read schema SQL file
        with open("schema_snowflake.sql", "r") as f:
            schema_sql = f.read()
            
        # Split by semicolon and execute each statement
        statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
        
        for statement in statements:
            try:
                cursor.execute(statement)
                print(f"Executed: {statement[:60]}...")
            except Exception as e:
                print(f"Warning: {e}")
        
        conn.commit()
        cursor.close()
        conn.close()
        print(f"Database initialized in Snowflake successfully.")
        print(f"Database: {SNOWFLAKE_CONFIG['database']}")
        print(f"Schema: {SNOWFLAKE_CONFIG['schema']}")
    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

if __name__ == "__main__":
    init_db()
