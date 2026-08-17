
import os
from datetime import date, datetime
from contextlib import contextmanager

import streamlit as st
import snowflake.connector
from dotenv import load_dotenv

load_dotenv()

SF_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
SF_USER = os.getenv("SNOWFLAKE_USER")
SF_PASSWORD = os.getenv("SNOWFLAKE_TOKEN")
SF_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
SF_DATABASE = os.getenv("SNOWFLAKE_DATABASE", "DOG_WHISPERER")
SF_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC")


@st.cache_resource
def get_connection():
    """Return a cached connection to Snowflake."""
    conn = snowflake.connector.connect(
        account=SF_ACCOUNT,
        user=SF_USER,
        password=SF_PASSWORD,
        warehouse=SF_WAREHOUSE,
    )
    # Set up database/schema once when connection is created
    cur = conn.cursor()
    cur.execute(f"USE DATABASE {SF_DATABASE}")
    cur.execute(f"USE SCHEMA {SF_SCHEMA}")
    cur.close()
    return conn


def init_schema() -> None:
    """Creates the database/schema/tables if they don't already exist.
    Safe to call every app startup — all statements are idempotent."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS {SF_DATABASE}")
    cur.execute(f"CREATE SCHEMA IF NOT EXISTS {SF_DATABASE}.{SF_SCHEMA}")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS pets (
            pet_id       STRING DEFAULT UUID_STRING(),
            owner_id     STRING NOT NULL DEFAULT 'anonymous',
            name         STRING NOT NULL,
            species      STRING DEFAULT 'dog',
            breed        STRING,
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
            PRIMARY KEY (pet_id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS meal_logs (
            log_id       STRING DEFAULT UUID_STRING(),
            pet_id       STRING NOT NULL,
            food_item    STRING NOT NULL,
            amount_grams FLOAT,
            notes        STRING,
            logged_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
            PRIMARY KEY (log_id),
            FOREIGN KEY (pet_id) REFERENCES pets (pet_id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS weight_logs (
            log_id       STRING DEFAULT UUID_STRING(),
            pet_id       STRING NOT NULL,
            weight_kg    FLOAT NOT NULL,
            logged_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
            PRIMARY KEY (log_id),
            FOREIGN KEY (pet_id) REFERENCES pets (pet_id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS walk_logs (
            log_id       STRING DEFAULT UUID_STRING(),
            pet_id       STRING NOT NULL,
            duration_min FLOAT NOT NULL,
            notes        STRING,
            logged_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
            PRIMARY KEY (log_id),
            FOREIGN KEY (pet_id) REFERENCES pets (pet_id)
        )
    """)

    conn.commit()


@st.cache_data
def list_pets(owner_id: str) -> list:
    """Retrieve all pets for an owner. Cached per owner_id."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT pet_id, name, species, breed FROM pets WHERE owner_id = %s", (owner_id,))
    return [{"pet_id": r[0], "name": r[1], "species": r[2], "breed": r[3]} for r in cur.fetchall()]


def add_pet(owner_id: str, name: str, species: str = "dog", breed: str = None) -> None:
    """Add a new pet. Clears the pets cache for this owner."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO pets (owner_id, name, species, breed) VALUES (%s, %s, %s, %s)",
        (owner_id, name, species, breed),
    )
    conn.commit()
    # Clear cached pets list for this owner
    list_pets.clear()


def log_meal(pet_id: str, food_item: str, amount_grams: float = None, notes: str = None) -> None:
    """Log a meal for a pet. Clears related caches."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO meal_logs (pet_id, food_item, amount_grams, notes) VALUES (%s, %s, %s, %s)",
        (pet_id, food_item, amount_grams, notes),
    )
    conn.commit()
    # Clear caches for this pet's data
    get_meal_logs.clear()
    daily_food_total.clear()


def log_weight(pet_id: str, weight_kg: float) -> None:
    """Log a weight measurement for a pet. Clears related caches."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO weight_logs (pet_id, weight_kg) VALUES (%s, %s)",
        (pet_id, weight_kg),
    )
    conn.commit()
    # Clear cache for this pet's weight trend
    get_weight_trend.clear()


def log_walk(pet_id: str, duration_min: float, notes: str = None) -> None:
    """Log a walk for a pet. Clears related caches."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO walk_logs (pet_id, duration_min, notes) VALUES (%s, %s, %s)",
        (pet_id, duration_min, notes),
    )
    conn.commit()
    # Clear cache for this pet's walk trends
    get_walk_trend.clear()
    daily_walk_total.clear()
    get_walk_logs.clear()


@st.cache_data
def get_weight_trend(pet_id: str, days: int = 30) -> list:
    """Get weight log history. Cached per pet_id and days."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT logged_at, weight_kg FROM weight_logs
        WHERE pet_id = %s AND logged_at >= DATEADD(day, -%s, CURRENT_TIMESTAMP())
        ORDER BY logged_at ASC
        """,
        (pet_id, days),
    )
    return [{"logged_at": r[0], "weight_kg": r[1]} for r in cur.fetchall()]


@st.cache_data
def get_meal_logs(pet_id: str, days: int = 7) -> list:
    """Get recent meal logs. Cached per pet_id and days."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT logged_at, food_item, amount_grams, notes FROM meal_logs
        WHERE pet_id = %s AND logged_at >= DATEADD(day, -%s, CURRENT_TIMESTAMP())
        ORDER BY logged_at DESC
        """,
        (pet_id, days),
    )
    return [
        {"logged_at": r[0], "food_item": r[1], "amount_grams": r[2], "notes": r[3]}
        for r in cur.fetchall()
    ]


@st.cache_data
def daily_food_total(pet_id: str, days: int = 7) -> list:
    """Get daily food totals for charting. Cached per pet_id and days."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            DATE(logged_at) AS log_date,
            SUM(amount_grams) AS total_grams
        FROM meal_logs
        WHERE pet_id = %s AND logged_at >= DATEADD(day, -%s, CURRENT_TIMESTAMP())
        GROUP BY DATE(logged_at)
        ORDER BY log_date ASC
        """,
        (pet_id, days),
    )
    return [{"log_date": r[0], "total_grams": r[1]} for r in cur.fetchall()]


@st.cache_data
def get_walk_logs(pet_id: str, days: int = 7) -> list:
    """Get recent walk logs. Cached per pet_id and days."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT logged_at, duration_min, notes FROM walk_logs
        WHERE pet_id = %s AND logged_at >= DATEADD(day, -%s, CURRENT_TIMESTAMP())
        ORDER BY logged_at DESC
        """,
        (pet_id, days),
    )
    return [
        {"logged_at": r[0], "duration_min": r[1], "notes": r[2]}
        for r in cur.fetchall()
    ]


@st.cache_data
def get_walk_trend(pet_id: str, days: int = 30) -> list:
    """Get walk history over time. Cached per pet_id and days."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT logged_at, duration_min FROM walk_logs
        WHERE pet_id = %s AND logged_at >= DATEADD(day, -%s, CURRENT_TIMESTAMP())
        ORDER BY logged_at ASC
        """,
        (pet_id, days),
    )
    return [{"logged_at": r[0], "duration_min": r[1]} for r in cur.fetchall()]


@st.cache_data
def daily_walk_total(pet_id: str, days: int = 7) -> list:
    """Get daily walk totals for charting. Cached per pet_id and days."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            DATE(logged_at) AS log_date,
            SUM(duration_min) AS total_duration,
            COUNT(*) AS walk_count
        FROM walk_logs
        WHERE pet_id = %s AND logged_at >= DATEADD(day, -%s, CURRENT_TIMESTAMP())
        GROUP BY DATE(logged_at)
        ORDER BY log_date ASC
        """,
        (pet_id, days),
    )
    return [{"log_date": r[0], "total_duration": r[1], "walk_count": r[2]} for r in cur.fetchall()]
