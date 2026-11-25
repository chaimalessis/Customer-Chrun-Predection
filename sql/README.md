# SQL analysis

15 SQLite queries on the scored customer table. They reproduce the KPIs used in the report and the Power BI dashboard.

## Database

- Engine: SQLite
- Table: `churn`, loaded from `data/processed/churn_scored.csv` (2,850 rows, one per customer)
- Key columns: `CustomerID`, `Churn`, `Customer_Value`, `Churn_Probability`, `Value_At_Risk`, `Risk_Band`, `Value_Tier`, `Retention_Action`, `Expected_Retention_Benefit`, `Contact_Recommended`, plus the usage variables and labels.

## How to run

```bash
python sql/create_database.py     # builds sql/churn_analysis.db and prints the result of all 15 queries
```

Or open `churn_analysis.db` in any SQLite client (DB Browser for SQLite, DBeaver, VS Code) and run `churn_analysis.sql`.

## Queries

| # | Question |
|---|---|
| 1 | Headline KPIs: customers, churn rate, value at risk, customers to contact |
| 2-4 | Churn rate by tariff plan, complaints and status |
| 5 | Risk band summary: customers and value at risk |
| 6 | Retention action plan: customers, value at risk, expected benefit |
| 7 | Top 20 customers to contact first |
| 8 | Churn by subscription length |
| 9-13 | Churn by call failures, frequency of use, SMS, seconds of use and customer value |
| 14 | Data quality: duplicate check (customer IDs unique) |
| 15 | Outlier check: churn of the top 1% of users by usage |
