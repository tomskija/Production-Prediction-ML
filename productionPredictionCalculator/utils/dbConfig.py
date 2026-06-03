############################################################################################
import os
import sqlite3
import pandas as pd
############################################################################################
try:
    import psycopg2
    from psycopg2.extras import execute_values
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
############################################################################################


############################################################################################
# --- Schema ---
SCHEMA_WELL_DATA = """
    CREATE TABLE IF NOT EXISTS well_data (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id      TEXT,
        Por         REAL,
        Brittle     REAL,
        Latitude    REAL,
        Longitude   REAL,
        Production  REAL
    );
"""

SCHEMA_RUN_RESULTS = """
    CREATE TABLE IF NOT EXISTS run_results (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        mlflow_run_id        TEXT,
        timestamp            TEXT,
        status               TEXT,
        selection_mode       TEXT,
        selected_features    TEXT,
        best_sampling_method TEXT,
        max_depth            INTEGER,
        num_trees            INTEGER,
        max_features         INTEGER,
        split_seed           INTEGER,
        rf_seed              INTEGER,
        r2_test              REAL,
        rmse_test            REAL,
        mape_test            REAL,
        shap_ranking         TEXT
    );
"""

SCHEMA_WELL_DATA_PG = SCHEMA_WELL_DATA.replace('INTEGER PRIMARY KEY AUTOINCREMENT', 'SERIAL PRIMARY KEY')
SCHEMA_RUN_RESULTS_PG = SCHEMA_RUN_RESULTS.replace('INTEGER PRIMARY KEY AUTOINCREMENT', 'SERIAL PRIMARY KEY')

############################################################################################
class DBConfig:
    """
    Handles SQLite (local) and PostgreSQL (production) connections and operations.
    DB type is determined by environment — localTesing=True uses SQLite, else PostgreSQL.
    """
    ########################################################################################
    def __init__(self, localTesing=False):
        self.localTesing = localTesing
        self.conn        = None
        if localTesing:
            self._connect_sqlite()
        else:
            self._connect_postgres()

    ########################################################################################
    def _connect_sqlite(self):
        db_path  = os.path.join(os.path.dirname(__file__), '..', 'Data', 'production.db')
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema_sqlite()
        print(f"SQLite connected : {os.path.abspath(db_path)}")

    ########################################################################################
    def _connect_postgres(self):
        if not PSYCOPG2_AVAILABLE:
            raise ImportError("psycopg2 not installed — required for PostgreSQL connection")
        self.conn = psycopg2.connect(
            host     = os.environ.get('POSTGRES_HOST',     'postgres'),
            port     = os.environ.get('POSTGRES_PORT',     '5432'),
            dbname   = os.environ.get('POSTGRES_DB',       'production_prediction'),
            user     = os.environ.get('POSTGRES_USER',     'admin'),
            password = os.environ.get('POSTGRES_PASSWORD', 'admin'),
        )
        self._init_schema_postgres()
        print(f"PostgreSQL connected : {os.environ.get('POSTGRES_HOST', 'postgres')}")

    ########################################################################################
    def _init_schema_sqlite(self):
        cur = self.conn.cursor()
        cur.execute(SCHEMA_WELL_DATA)
        cur.execute(SCHEMA_RUN_RESULTS)
        self.conn.commit()

    ########################################################################################
    def _init_schema_postgres(self):
        cur = self.conn.cursor()
        cur.execute(SCHEMA_WELL_DATA_PG)
        cur.execute(SCHEMA_RUN_RESULTS_PG)
        self.conn.commit()

    ########################################################################################
    def load_well_data(self):
        """Load well data from DB into a DataFrame."""
        return pd.read_sql("SELECT * FROM well_data", self.conn)

    ########################################################################################
    def insert_well_data(self, df=pd.DataFrame(), run_id=''):
        """Insert well data DataFrame into DB."""
        df = df.copy()
        df['run_id'] = run_id
        df.to_sql('well_data', self.conn, if_exists='append', index=False) if self.localTesing else self._insert_postgres(df=df, table='well_data')
        print(f"Inserted {len(df)} rows into well_data")

    ########################################################################################
    def insert_run_results(self, results={}):
        """Insert a run results dict into DB."""
        cur = self.conn.cursor()
        cols = ', '.join(results.keys())
        if self.localTesing:
            placeholders = ', '.join(['?' for _ in results])
            cur.execute(f"INSERT INTO run_results ({cols}) VALUES ({placeholders})", list(results.values()))
        else:
            placeholders = ', '.join(['%s' for _ in results])
            cur.execute(f"INSERT INTO run_results ({cols}) VALUES ({placeholders})", list(results.values()))
        self.conn.commit()
        print(f"Run results inserted : {results.get('mlflow_run_id', '')}")

    ########################################################################################
    def _insert_postgres(self, df=pd.DataFrame(), table=''):
        """Bulk insert DataFrame into PostgreSQL table."""
        cur     = self.conn.cursor()
        cols    = ', '.join(df.columns)
        values  = [tuple(row) for row in df.itertuples(index=False)]
        execute_values(cur, f"INSERT INTO {table} ({cols}) VALUES %s", values)
        self.conn.commit()

    ########################################################################################
    def close(self):
        if self.conn: self.conn.close()
############################################################################################