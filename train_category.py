"""
Trainer/runner for category classifier.

Usage:
  python train_category.py --snowflake-conn '<conn-string>'

You can also import `train_category_model` from `operational_recommendations` and call it
with a Snowflake connection object.
"""
import argparse
import os
import logging
from operational_recommendations import train_category_model
import snowflake.connector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--user', help='Snowflake user')
    p.add_argument('--password', help='Snowflake password')
    p.add_argument('--account', help='Snowflake account')
    p.add_argument('--warehouse', help='Snowflake warehouse', default=None)
    p.add_argument('--database', help='Snowflake database', default=None)
    p.add_argument('--schema', help='Snowflake schema', default=None)
    p.add_argument('--model-path', help='Path to save model', default='models/category_classifier.joblib')
    return p.parse_args()


def main():
    args = parse_args()
    conn_kwargs = {}
    if args.user and args.password and args.account:
        conn_kwargs.update(user=args.user, password=args.password, account=args.account)
    else:
        # try environment variables
        conn_kwargs.update({
            'user': os.getenv('SF_USER'),
            'password': os.getenv('SF_PASSWORD'),
            'account': os.getenv('SF_ACCOUNT'),
            'warehouse': os.getenv('SF_WAREHOUSE'),
            'database': os.getenv('SF_DATABASE'),
            'schema': os.getenv('SF_SCHEMA'),
        })

    # override with explicit args
    if args.warehouse:
        conn_kwargs['warehouse'] = args.warehouse
    if args.database:
        conn_kwargs['database'] = args.database
    if args.schema:
        conn_kwargs['schema'] = args.schema

    logger.info('Connecting to Snowflake...')
    conn = snowflake.connector.connect(**{k: v for k, v in conn_kwargs.items() if v})
    logger.info('Connected')

    summary = train_category_model(conn, model_path=args.model_path)
    logger.info('Training complete: %s', summary)
    conn.close()


if __name__ == '__main__':
    main()
