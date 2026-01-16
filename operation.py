"""
Database operations for operational recommendations and suggestions.
Handles both Snowflake (anomalies) and PostgreSQL (operation suggestions) storage.
"""

import json
import logging
import os
import subprocess
import shlex
import re
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
        raw = content
        try:
            if "```json" in raw:
                raw = raw.split("```json", 1)[1].rsplit("```", 1)[0]
            elif "```" in raw:
                raw = raw.split("```", 1)[1].rsplit("```", 1)[0]

            # Attempt direct parse first
            try:
                suggestion_data = json.loads(raw.strip())
                return suggestion_data
            except Exception as e_direct:
                # Try to extract the first JSON object in the text
                import re as _re
                m = _re.search(r"\{.*\}", raw, _re.DOTALL)
                if m:
                    candidate = m.group(0)
                    # Fix common issues: trailing commas
                    candidate_fixed = _re.sub(r",\s*\}", "}", candidate)
                    candidate_fixed = _re.sub(r",\s*\]", "]", candidate_fixed)
                    try:
                        suggestion_data = json.loads(candidate_fixed)
                        return suggestion_data
                    except Exception as e_candidate:
                        logger.warning(f"Failed to parse candidate JSON from OpenAI (attempts): {e_direct}; {e_candidate}")
                else:
                    logger.warning("No JSON object found in OpenAI response; falling back")

        except Exception as e_clean:
            logger.warning(f"Error cleaning OpenAI content: {e_clean}")

        # If parsing failed, log raw content for debugging and return a safe fallback
        logger.error(f"Failed to parse OpenAI response into JSON. Raw content:\n{content}")
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
                "detailed_analysis": f"AI output could not be parsed as JSON. See logs."
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


def infer_action_category_priority(well_id: str, production_data: dict = None, sensor_metrics: dict = None, hint: str = None) -> dict:
    """Use OpenAI to infer a concise `action`, `category` and `priority` for a well given data.

    Returns: {'action': str, 'category': 'Production'|'Optimization'|'Maintenance', 'priority': 'HIGH'|'MEDIUM'|'LOW'}
    """
    try:
        # Build compact context
        context = f"Well: {well_id}\n"
        if production_data:
            context += "Production:\n"
            for k, v in (production_data or {}).items():
                context += f" - {k}: {v}\n"
        if sensor_metrics:
            context += "Sensors:\n"
            for k, v in (sensor_metrics or {}).items():
                context += f" - {k}: {v}\n"
        if hint:
            context += f"\nModel hint: {hint}\n"

        prompt = f"""You are a petroleum engineer. Given the well context, choose a single concise recommended action (short phrase), a high-level category from [Production, Optimization, Maintenance], and a priority from [HIGH, MEDIUM, LOW]. Respond as JSON with keys: action, category, priority.

    Preferred action style: use concise, operational phrasing similar to these examples:
    - "Increase ESP frequency from 55Hz to 58Hz"
    - "Increase gas injection rate from 2.5 MMSCF to 3.2 MMSCF"
    - "Replace impeller and perform alignment check within 7 days"
    - "Reduce separator pressure from 125 PSI to 110 PSI"
    - "Reduce pumping speed from 12 SPM to 10 SPM immediately"

    Context:
    {context}

    Example output:
    {{"action": "Increase ESP frequency from 55Hz to 58Hz", "category": "Production", "priority": "HIGH"}}
    """

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert petroleum engineer. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        content = response.choices[0].message.content
        # strip code fences
        if "```" in content:
            content = content.split("```")[-1].strip()
        parsed = json.loads(content)
        action = parsed.get('action')
        category = parsed.get('category')
        priority = parsed.get('priority')
        return {'action': action, 'category': category, 'priority': priority}
    except Exception as e:
        logger.warning(f"OpenAI inference failed, falling back to hint/mapping: {e}")
        # fallback: use hint or simple keyword mapping to pick a reasonable category
        fallback_action = (hint or 'Investigate well metrics')
        act_low = (fallback_action or '').lower()
        # simple keyword-based heuristics
        if any(k in act_low for k in ('increase', 'esp', 'pump', 'choke', 'polish')):
            fallback_category = 'Production'
        elif any(k in act_low for k in ('optimi', 'reduce', 'chemical', 'cost', 'efficien')):
            fallback_category = 'Optimization'
        elif any(k in act_low for k in ('inspect', 'maintenance', 'fail', 'fault', 'cable', 'replace')):
            fallback_category = 'Maintenance'
        else:
            fallback_category = 'Production'
        fallback_priority = 'MEDIUM'
        return {'action': fallback_action, 'category': fallback_category, 'priority': fallback_priority}


def refine_suggestion_with_openai(well_id: str, suggested_action: str, production_data: dict = None,
                                  sensor_metrics: dict = None, model_confidence: float = None,
                                  category_hint: str = None) -> dict:
    """
    Ask OpenAI to refine or confirm the action label, pick a category, and suggest a priority
    based on the provided metrics and model confidence.

    Returns: {'action': str, 'category': str, 'priority': 'HIGH'|'MEDIUM'|'LOW'}
    Falls back to the suggested_action and defaults if the API call fails.
    """
    try:
        # Build a concise prompt
        context = f"Well: {well_id}\nSuggested action: {suggested_action}\nModel confidence: {model_confidence}\n"
        if production_data:
            context += "\nProduction:\n"
            for k, v in (production_data or {}).items():
                context += f" - {k}: {v}\n"
        if sensor_metrics:
            context += "\nSensors:\n"
            for k, v in (sensor_metrics or {}).items():
                context += f" - {k}: {v}\n"

        prompt = f"""
    You are a petroleum engineer advisor. Given the suggested action and recent production/sensor metrics, decide whether to keep or adjust the action label, assign one of the categories [Production, Optimization, Maintenance], and choose a priority [HIGH, MEDIUM, LOW].

    Return a JSON object with keys: action, category, priority. Keep values concise. Use operational phrasing like:
    - "Increase ESP frequency from 55Hz to 58Hz"
    - "Increase gas injection rate from 2.5 MMSCF to 3.2 MMSCF"
    - "Replace impeller and perform alignment check within 7 days"
    - "Reduce separator pressure from 125 PSI to 110 PSI"
    - "Reduce pumping speed from 12 SPM to 10 SPM immediately"

    Context:
    {context}

    Respond with JSON only.
    """

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a concise technical advisor. Respond with JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=200
        )

        content = response.choices[0].message.content
        # strip markdown if present
        if '```' in content:
            content = content.split('```')[-2].strip()
        obj = json.loads(content)
        # sanitize and validate category
        action = obj.get('action') or suggested_action
        category = obj.get('category') or ''
        priority = obj.get('priority') or ('HIGH' if (model_confidence or 0) > 0.85 else 'MEDIUM')
        valid_cats = {'Production', 'Optimization', 'Maintenance'}
        if category not in valid_cats:
            # fallback to provided category_hint or default to Production
            category = category_hint if category_hint in valid_cats else 'Production'
        return {'action': action, 'category': category, 'priority': priority}
    except Exception as e:
        logger.warning(f"Refine OpenAI call failed: {e}")
        fallback_cat = category_hint if category_hint in {'Production', 'Optimization', 'Maintenance'} else 'Production'
        return {'action': suggested_action, 'category': fallback_cat, 'priority': ('HIGH' if (model_confidence or 0) > 0.85 else 'MEDIUM')}


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
        # helper: truncate long strings to fit schema limits
        def _t(s, n):
            try:
                if s is None:
                    return None
                s2 = str(s)
                return s2 if len(s2) <= n else s2[:n]
            except Exception:
                return s

        # Refine action/category/priority first (may override model hint)
        try:
            # Pass the current ML-derived `category` as a hint so OpenAI can consider it,
            # but do not let OpenAI silently overwrite the ML category unless it
            # returns one of the canonical categories.
            refined = refine_suggestion_with_openai(
                well_id,
                action,
                production_data,
                sensor_metrics,
                model_confidence=(confidence / 100.0 if confidence and confidence > 1 else confidence),
                category_hint=category
            )
            # use refined values if returned; preserve ML category unless OpenAI returns a valid one
            if isinstance(refined, dict):
                action = refined.get('action') or action
                priority = refined.get('priority') or priority
                ref_cat = refined.get('category')
                if ref_cat in {'Production', 'Optimization', 'Maintenance'}:
                    category = ref_cat
        except Exception:
            logger.warning("Refine suggestion step failed; continuing with provided action/category")

        # Generate detailed suggestion using OpenAI (use refined action)
        logger.info(f"Generating detailed suggestion for {well_id} (action={action}, category={category})...")
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
            _t(category, 50),  # category
            _t(action, 255),  # action
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
            _t(op_details.get('current_performance', ''), 255),
            _t(op_details.get('optimal_performance', ''), 255),
            _t(op_details.get('detailed_analysis', ''), 2000),
            
            # Alternative models
            _t(alt_models.get('conservative_approach', ''), 2000),
            _t(alt_models.get('aggressive_optimization', ''), 2000),
            _t(alt_models.get('hybrid_model', ''), 2000),
            _t(alt_models.get('recommended_model', ''), 50),
            
            # Citations
            json.dumps(citations.get('referenced_data', [])),
            json.dumps(citations.get('referenced_data', [])),
            _t(citations.get('supporting_analysis', ''), 2000),
            _t(citations.get('data_quality', ''), 100),
            
            # Additional
            _t(reason, 2000),
            _t(expected_impact, 500),
            json.dumps({'production': production_data or {}, 'sensors': sensor_metrics or {}}),
            datetime.utcnow()
        )

        cur.execute(insert_sql, values)
        conn.commit()
        logger.info(f"✓ Saved operation suggestion for {well_id}: {action}")

        # Optional: insert alternative perspectives for inspection when enabled
        try:
            diversify = os.getenv('OPERATION_SUGGESTION_DIVERSIFY', 'false').lower() in ('1', 'true', 'yes')
            if diversify:
                alt_pairs = [
                    ('Optimization', f'Optimize settings for {action}'),
                    ('Maintenance', f'Inspect equipment related to {action}')
                ]
                for alt_cat, alt_action in alt_pairs:
                    alt_values = (
                        str(uuid4()),
                        well_id,
                        alt_cat,
                        alt_action,
                        status,
                        'MEDIUM',
                        float(confidence or 0.0),
                        float(prod_impact.get('increase_bbl_day', 0)),
                        float(prod_impact.get('increase_percent', 0)),
                        float(fin_impact.get('daily_expense_benefit_usd', 0)),
                        float(fin_impact.get('implementation_cost_usd', 0)),
                        float(fin_impact.get('net_daily_benefit_usd', 0)),
                        float(fin_impact.get('asset_value_increase_usd', 0)),
                        float(op_details.get('time_reduced_hours', 0)),
                        op_details.get('current_performance', ''),
                        op_details.get('optimal_performance', ''),
                        op_details.get('detailed_analysis', ''),
                        alt_models.get('conservative_approach', ''),
                        alt_models.get('aggressive_optimization', ''),
                        alt_models.get('hybrid_model', ''),
                        alt_models.get('recommended_model', ''),
                        json.dumps(citations.get('referenced_data', [])),
                        json.dumps(citations.get('referenced_data', [])),
                        citations.get('supporting_analysis', ''),
                        citations.get('data_quality', ''),
                        reason,
                        expected_impact,
                        json.dumps({'production': production_data or {}, 'sensors': sensor_metrics or {}}),
                        datetime.utcnow()
                    )
                    cur.execute(insert_sql, alt_values)
                conn.commit()
                logger.info(f"✓ Inserted alternative perspectives for {well_id}")
        except Exception:
            logger.warning("Failed to insert alternative perspectives; continuing")

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


def stream_run_operational_pipeline(confidence: str = "0.2"):
    """Spawn the operational pipeline script and stream logs as Server-Sent Events (SSE).

    Yields SSE `data: ...\n\n` chunks containing JSON objects with either `line`,
    or `status`/`returncode`/`error` fields.
    """
    process = None
    try:
        venv_py = os.path.join(os.getcwd(), '.venv', 'bin', 'python')
        if not os.path.exists(venv_py):
            venv_py = 'python'

        script_path = os.path.join(os.getcwd(), 'run_operational_pipeline.py')
        cmd = [venv_py, script_path, '--confidence', str(confidence)]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        )

        # Patterns we consider 'important'
        processing_re = re.compile(r"Processing:\s*(?P<well_id>[^-]+)\s*-\s*(?P<action>.+?)\s*\(confidence:\s*(?P<conf>[\d%.]+)\)")
        saved_postgres_re = re.compile(r"✓ Saved to PostgreSQL")
        failed_save_re = re.compile(r"✗ Failed to save|Failed to save to PostgreSQL|Failed to save")
        predictions_re = re.compile(r"✓ Predictions completed:\s*(?P<count>\d+) recommendations detected")
        train_acc_re = re.compile(r"Train Accuracy: (.+)")
        summary_key_re = re.compile(r"^(Recommendations Processed:|Saved to Snowflake:|Saved to PostgreSQL:|Failed:)\s*(?P<num>\d+)")
        detail_line_re = re.compile(r"\s*(?P<well_id>\S+)\s*\|\s*(?P<action>.+?)\s*\|\s*(?P<conf>[\d%.]+)\s*\|\s*SF:(?P<sf>.)\s*PG:(?P<pg>.)")

        last_ctx = {'well_id': None, 'action': None}

        if process.stdout is not None:
            for raw_line in iter(process.stdout.readline, ''):
                if raw_line is None:
                    break
                line = raw_line.rstrip('\n').strip()
                if not line:
                    continue

                # Error or traceback: always surface
                if 'ERROR' in line or 'Traceback' in line or 'Exception' in line:
                    payload = json.dumps({'type': 'error', 'message': line})
                    yield f"data: {payload}\n\n"
                    continue

                m = processing_re.search(line)
                if m:
                    well = m.group('well_id').strip()
                    action = m.group('action').strip()
                    conf = m.group('conf').strip()
                    last_ctx['well_id'] = well
                    last_ctx['action'] = action
                    payload = json.dumps({'type': 'processing', 'well_id': well, 'action': action, 'confidence': conf})
                    yield f"data: {payload}\n\n"
                    continue

                if saved_postgres_re.search(line):
                    payload = json.dumps({'type': 'saved', 'well_id': last_ctx.get('well_id'), 'action': last_ctx.get('action'), 'message': 'saved_postgres'})
                    yield f"data: {payload}\n\n"
                    continue

                if failed_save_re.search(line):
                    payload = json.dumps({'type': 'failed_save', 'well_id': last_ctx.get('well_id'), 'action': last_ctx.get('action'), 'message': line})
                    yield f"data: {payload}\n\n"
                    continue

                m2 = predictions_re.search(line)
                if m2:
                    payload = json.dumps({'type': 'predictions', 'count': int(m2.group('count'))})
                    yield f"data: {payload}\n\n"
                    continue

                if 'PIPELINE SUMMARY' in line or 'OPERATIONAL RECOMMENDATIONS PIPELINE' in line or 'Processing Recommendations for Database Storage' in line:
                    payload = json.dumps({'type': 'stage', 'message': line})
                    yield f"data: {payload}\n\n"
                    continue

                m3 = train_acc_re.search(line)
                if m3:
                    payload = json.dumps({'type': 'train_summary', 'message': m3.group(1).strip()})
                    yield f"data: {payload}\n\n"
                    continue

                m4 = detail_line_re.search(line)
                if m4:
                    payload = json.dumps({'type': 'detail', 'well_id': m4.group('well_id'), 'action': m4.group('action').strip(), 'confidence': m4.group('conf'), 'sf': m4.group('sf'), 'pg': m4.group('pg')})
                    yield f"data: {payload}\n\n"
                    continue

                m5 = summary_key_re.search(line)
                if m5:
                    key = line.split(':')[0].strip()
                    num = int(m5.group('num'))
                    payload = json.dumps({'type': 'summary', 'key': key, 'value': num})
                    yield f"data: {payload}\n\n"
                    continue

                # By default, only emit short informational lines that are concise (e.g., lines starting with ✓ or ✗)
                if line.startswith('✓') or line.startswith('✗') or line.startswith('Start Time') or line.startswith('Confidence Threshold') or line.startswith('Train') or line.startswith('No high-confidence'):
                    payload = json.dumps({'type': 'info', 'message': line})
                    yield f"data: {payload}\n\n"
                    continue

                # skip verbose lines

        rc = process.wait() if process is not None else 0
        yield f"data: {json.dumps({'type': 'finished', 'returncode': rc})}\n\n"

    except GeneratorExit:
        try:
            if process:
                process.terminate()
        except Exception:
            pass
        raise
    except Exception as e:
        try:
            if process:
                process.kill()
        except Exception:
            pass
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
