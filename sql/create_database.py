"""Build sql/churn_analysis.db from the scored dataset and run the 15 queries in churn_analysis.sql."""
import re
import sqlite3
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = HERE.parent / "data" / "processed" / "churn_scored.csv"
DB = HERE / "churn_analysis.db"
SQL = HERE / "churn_analysis.sql"

con = sqlite3.connect(DB)
pd.read_csv(CSV).to_sql("churn", con, if_exists="replace", index=False)
print(f"Table 'churn' created in {DB.name}: {con.execute('SELECT COUNT(*) FROM churn').fetchone()[0]} rows\n")

# split the file into numbered queries ("-- 1. title" ... ";")
text = SQL.read_text()
queries = re.findall(r"--\s*(\d+)\.\s*(.+?)\n(.*?;)\s*(?=--|\Z)", text, flags=re.S)
for num, title, query in queries:
    print(f"--- Query {num}: {title.strip()}")
    print(pd.read_sql_query(query, con).to_string(index=False), "\n")
con.close()
