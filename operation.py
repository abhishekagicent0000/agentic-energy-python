"""
Database operations for operational recommendations and suggestions.
Handles both Snowflake (anomalies) and PostgreSQL (operation suggestions) storage.
"""

import json
import logging
import os
from datetime import datetime
from uuid import uuid4
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import openai

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# OpenAI Configuration
api_key = os.getenv("OPENAI_API_KEY")
APP_URL = os.getenv("APP_URL", "http://localhost:5000")

if not api_key:
    logger.error("OPENAI_API_KEY is missing in .env")

client = openai.OpenAI(
    api_key=api_key,
    base_url=os.getenv("OPENAI_API_BASE"),
    default_headers={
        "HTTP-Referer": APP_URL,
        "X-Title": "Well Anomaly Detection",
    }
)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "openai/gpt-3.5-turbo")

# PostgreSQL Configuration
DATABASE_URL = os.getenv("DATABASE_URL")


def get_pg_connection():
    """Get PostgreSQL connection."""
    try:
        if not DATABASE_URL:
            logger.error("DATABASE_URL is not set in .env")
            return None
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(DATABASE_URL)
        clean_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, '', parsed.fragment))
        conn = psycopg2.connect(clean_url)
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}")
        return None


def create_operation_suggestion_table():
    """Create operation_suggestion table in PostgreSQL."""
    conn = get_pg_connection()
    if not conn:
        logger.error("Could not create table: no PostgreSQL connection")
        return False
    
    cur = conn.cursor()
    try:
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS operation_suggestion (
            id VARCHAR(50) PRIMARY KEY,
            well_id VARCHAR(50) NOT NULL,
            category VARCHAR(50) DEFAULT 'Production',
            action VARCHAR(255) NOT NULL,
            status VARCHAR(50) DEFAULT 'New',
            priority VARCHAR(50) DEFAULT 'HIGH',
            confidence_percent FLOAT,
            
            -- Production Impact
            production_increase_bbl_day FLOAT,
            production_increase_percent FLOAT,
            
            -- Financial Impact
            daily_expense_benefit_usd FLOAT,
            implementation_cost_usd FLOAT,
            net_daily_benefit_usd FLOAT,
            asset_value_increase_usd FLOAT,
            
            -- Operational Details
            time_reduced_hours FLOAT,
            detailed_analysis TEXT,
            current_performance VARCHAR(255),
            optimal_performance VARCHAR(255),
            
            -- Alternative Models
            conservative_approach TEXT,
            aggressive_optimization TEXT,
            hybrid_model TEXT,
            recommended_model VARCHAR(50),
            
            -- Citations and References
            citations TEXT,
            referenced_data TEXT,
            supporting_analysis TEXT,
            data_quality VARCHAR(100),
            
            -- Additional Fields
            reason TEXT,
            expected_impact VARCHAR(500),
            metrics_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cur.execute(create_table_sql)
        conn.commit()
        logger.info("✓ operation_suggestion table created/verified")
        return True
    except Exception as e:
        logger.error(f"Error creating table: {e}")
        return False
    finally:
        cur.close()
        conn.close()


def generate_operation_suggestion_with_openai(well_id: str, action: str, 
                                             production_data: dict = None,
                                             sensor_metrics: dict = None) -> dict:
    """
    Generate detailed operation suggestion using OpenAI.
    
    Args:
        well_id: Well identifier
        action: Recommended action (e.g., "Increase ESP frequency")
        production_data: Dict with current production metrics
        sensor_metrics: Dict with sensor readings
    
    Returns:
        Dict with detailed suggestion data
    """
    try:
        # Build context from available data
        context = f"Well: {well_id}\nRecommended Action: {action}\n"
        
        if production_data:
            context += f"\nProduction Data:\n"
            for key, value in production_data.items():
                context += f"  - {key}: {value}\n"
        
        if sensor_metrics:
            context += f"\nSensor Metrics:\n"
            for key, value in sensor_metrics.items():
                context += f"  - {key}: {value}\n"

        prompt = f"""You are an expert petroleum engineer and reservoir analyst. Generate a detailed operational recommendation for an oil well.

{context}

Provide a comprehensive analysis in the following JSON format:
{{
    "production_impact": {{
        "increase_bbl_day": <number>,
        "increase_percent": <number>,
        "confidence_percent": <number>
    }},
    "financial_impact": {{
        "daily_expense_benefit_usd": <number>,
        "implementation_cost_usd": <number>,
        "net_daily_benefit_usd": <number>,
        "asset_value_increase_usd": <number>
    }},
    "operational_details": {{
        "time_reduced_hours": <number>,
        "current_performance": "<description>",
        "optimal_performance": "<description>",
        "detailed_analysis": "<detailed technical analysis>"
    }},
    "alternative_models": {{
        "conservative_approach": "{{approach description, expected uplift}}",
        "aggressive_optimization": "{{approach description, expected uplift}}",
        "hybrid_model": "{{approach description, expected uplift}}",
        "recommended_model": "hybrid_model|conservative_approach|aggressive_optimization"
    }},
    "citations": {{
        "referenced_data": [
            "SPE Paper reference",
            "Company Internal Report",
            "Vendor Technical Manual",
            "Proprietary ML Model"
        ],
        "supporting_analysis": "<detailed supporting analysis with references>",
        "data_quality": "<High/Medium/Low confidence description>"
    }}
    ,
    "category": "Production|Optimization|Maintenance"
}}

Ensure all fields are present and the output is valid JSON."""

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a petroleum engineer. Output valid JSON only matching the required schema."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
        )

        content = response.choices[0].message.content
        
        # Clean markdown code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        suggestion_data = json.loads(content.strip())
        return suggestion_data

    except Exception as e:
        logger.error(f"Error generating suggestion: {e}")
        return {
            "production_impact": {
                "increase_bbl_day": 0,
                "increase_percent": 0,
                "confidence_percent": 0
            },
            "financial_impact": {
                "daily_expense_benefit_usd": 0,
                "implementation_cost_usd": 0,
                "net_daily_benefit_usd": 0,
                "asset_value_increase_usd": 0
            },
            "operational_details": {
                "time_reduced_hours": 0,
                "current_performance": "Unknown",
                "optimal_performance": "Unknown",
                "detailed_analysis": f"AI Generation failed: {str(e)}"
            },
            "alternative_models": {
                "conservative_approach": "N/A",
                "aggressive_optimization": "N/A",
                "hybrid_model": "N/A",
                "recommended_model": "N/A"
            },
            "citations": {
                "referenced_data": [],
                "supporting_analysis": "N/A",
                "data_quality": "Low"
            }
        }


def save_operation_suggestion(well_id: str, action: str, category: str = "Production", status: str = "New",
                             priority: str = "HIGH", confidence: float = 0.0,
                             production_data: dict = None, sensor_metrics: dict = None,
                             reason: str = "", expected_impact: str = "") -> bool:
    """
    Save operation suggestion to PostgreSQL with OpenAI-generated detailed content.
    
    Args:
        well_id: Well ID
        action: Recommended action
        status: Status (default: "New")
        priority: Priority level (default: "HIGH")
        confidence: Confidence percentage (0-100)
        production_data: Production metrics dict
        sensor_metrics: Sensor metrics dict
        reason: Reason for recommendation
        expected_impact: Expected impact description
    
    Returns:
        True if successful, False otherwise
    """
    conn = get_pg_connection()
    if not conn:
        logger.error("Could not save suggestion: no PostgreSQL connection")
        return False
    
    cur = conn.cursor()
    try:
        # Generate detailed suggestion using OpenAI
        logger.info(f"Generating detailed suggestion for {well_id}...")
        suggestion = generate_operation_suggestion_with_openai(
            well_id, action, production_data, sensor_metrics
        )

        # Extract data from OpenAI response
        prod_impact = suggestion.get('production_impact', {})
        fin_impact = suggestion.get('financial_impact', {})
        op_details = suggestion.get('operational_details', {})
        alt_models = suggestion.get('alternative_models', {})
        citations = suggestion.get('citations', {})

        # Prepare insert statement
        insert_sql = """
        INSERT INTO operation_suggestion (
            id, well_id, category, action, status, priority, confidence_percent,
            production_increase_bbl_day, production_increase_percent,
            daily_expense_benefit_usd, implementation_cost_usd, net_daily_benefit_usd,
            asset_value_increase_usd, time_reduced_hours,
            current_performance, optimal_performance, detailed_analysis,
            conservative_approach, aggressive_optimization, hybrid_model, recommended_model,
            citations, referenced_data, supporting_analysis, data_quality,
            reason, expected_impact, metrics_json, created_at
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s
        )
        """

        values = (
            str(uuid4()),  # id
            well_id,  # well_id
            category,  # category
            action,  # action
            status,  # status
            priority,  # priority
            float(confidence or 0.0),  # confidence_percent
            
            # Production impact
            float(prod_impact.get('increase_bbl_day', 0)),
            float(prod_impact.get('increase_percent', 0)),
            
            # Financial impact
            float(fin_impact.get('daily_expense_benefit_usd', 0)),
            float(fin_impact.get('implementation_cost_usd', 0)),
            float(fin_impact.get('net_daily_benefit_usd', 0)),
            float(fin_impact.get('asset_value_increase_usd', 0)),
            
            # Operational
            float(op_details.get('time_reduced_hours', 0)),
            op_details.get('current_performance', ''),
            op_details.get('optimal_performance', ''),
            op_details.get('detailed_analysis', ''),
            
            # Alternative models
            alt_models.get('conservative_approach', ''),
            alt_models.get('aggressive_optimization', ''),
            alt_models.get('hybrid_model', ''),
            alt_models.get('recommended_model', ''),
            
            # Citations
            json.dumps(citations.get('referenced_data', [])),
            json.dumps(citations.get('referenced_data', [])),
            citations.get('supporting_analysis', ''),
            citations.get('data_quality', ''),
            
            # Additional
            reason,
            expected_impact,
            json.dumps({'production': production_data or {}, 'sensors': sensor_metrics or {}}),
            datetime.utcnow()
        )

        cur.execute(insert_sql, values)
        conn.commit()
        logger.info(f"✓ Saved operation suggestion for {well_id}: {action}")
        return True

    except Exception as e:
        logger.error(f"Error saving operation suggestion: {e}", exc_info=True)
        conn.rollback()
        return False
    finally:
        cur.close()
        conn.close()


def get_operation_suggestions(well_id: str = None, status: str = None, 
                             priority: str = None, category: str = None, limit: int = 100) -> list:
    """
    Retrieve operation suggestions from PostgreSQL.
    
    Args:
        well_id: Filter by well ID (optional)
        status: Filter by status (optional)
        priority: Filter by priority (optional)
        limit: Maximum number of results
    
    Returns:
        List of suggestion dicts
    """
    conn = get_pg_connection()
    if not conn:
        logger.error("Could not retrieve suggestions: no PostgreSQL connection")
        return []
    
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        query = "SELECT * FROM operation_suggestion WHERE 1=1"
        params = []

        if well_id:
            query += " AND well_id = %s"
            params.append(well_id)
        
        if status:
            query += " AND status = %s"
            params.append(status)
        
        if priority:
            query += " AND priority = %s"
            params.append(priority)

        if category:
            query += " AND category = %s"
            params.append(category)

        query += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)

        cur.execute(query, params)
        results = cur.fetchall()
        
        # Convert RealDictRow to regular dicts
        return [dict(row) for row in results]

    except Exception as e:
        logger.error(f"Error retrieving suggestions: {e}")
        return []
    finally:
        cur.close()
        conn.close()


def update_operation_suggestion_status(suggestion_id: str, status: str) -> bool:
    """
    Update the status of an operation suggestion.
    
    Args:
        suggestion_id: UUID of the suggestion
        status: New status (e.g., "In Progress", "Completed", "Rejected")
    
    Returns:
        True if successful, False otherwise
    """
    conn = get_pg_connection()
    if not conn:
        logger.error("Could not update suggestion: no PostgreSQL connection")
        return False
    
    cur = conn.cursor()
    try:
        update_sql = """
        UPDATE operation_suggestion
        SET status = %s, updated_at = %s
        WHERE id = %s
        """
        cur.execute(update_sql, (status, datetime.utcnow(), suggestion_id))
        conn.commit()
        logger.info(f"✓ Updated suggestion {suggestion_id} status to {status}")
        return True

    except Exception as e:
        logger.error(f"Error updating suggestion: {e}")
        conn.rollback()
        return False
    finally:
        cur.close()
        conn.close()


def get_operation_suggestion_detail(suggestion_id: str) -> dict:
    """
    Get detailed information about a specific operation suggestion.
    
    Args:
        suggestion_id: UUID of the suggestion
    
    Returns:
        Dict with suggestion details
    """
    conn = get_pg_connection()
    if not conn:
        logger.error("Could not retrieve suggestion: no PostgreSQL connection")
        return {}
    
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute("SELECT * FROM operation_suggestion WHERE id = %s", (suggestion_id,))
        result = cur.fetchone()
        return dict(result) if result else {}

    except Exception as e:
        logger.error(f"Error retrieving suggestion detail: {e}")
        return {}
    finally:
        cur.close()
        conn.close()


if __name__ == '__main__':
    # Initialize table on module import if in main
    create_operation_suggestion_table()
    logger.info("✓ Operation module initialized")
