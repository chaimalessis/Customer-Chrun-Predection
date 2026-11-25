-- =====================================================================
-- Customer Churn Prediction: SQL KPI analysis (15 queries)
-- Engine : SQLite
-- Table  : churn  (loaded from data/processed/churn_scored.csv, 2,850 rows)
-- Build  : python sql/create_database.py   (creates sql/churn_analysis.db)
-- =====================================================================

-- 1. Headline KPIs
SELECT COUNT(*) AS customers, ROUND(100.0*AVG(Churn),1) AS churn_rate_pct,
       ROUND(SUM(Value_At_Risk),0) AS value_at_risk,
       SUM(Contact_Recommended='Yes') AS to_contact
FROM churn;
-- 2. Churn rate by tariff plan
SELECT Tariff_Plan_Label AS grp, COUNT(*) AS n, ROUND(100.0*AVG(Churn),1) AS churn_pct FROM churn GROUP BY 1 ORDER BY churn_pct DESC;
-- 3. Churn rate by complaints
SELECT Complains_Label AS grp, COUNT(*) AS n, ROUND(100.0*AVG(Churn),1) AS churn_pct FROM churn GROUP BY 1 ORDER BY churn_pct DESC;
-- 4. Churn rate by status
SELECT Status_Label AS grp, COUNT(*) AS n, ROUND(100.0*AVG(Churn),1) AS churn_pct FROM churn GROUP BY 1 ORDER BY churn_pct DESC;
-- 5. Risk band summary: customers and revenue at risk
SELECT Risk_Band, COUNT(*) AS customers, ROUND(SUM(Customer_Value),0) AS total_value, ROUND(SUM(Value_At_Risk),0) AS value_at_risk
FROM churn GROUP BY Risk_Band ORDER BY value_at_risk DESC;
-- 6. Retention action plan
SELECT Retention_Action, COUNT(*) AS customers, ROUND(SUM(Value_At_Risk),0) AS value_at_risk, ROUND(SUM(Expected_Retention_Benefit),0) AS expected_benefit
FROM churn GROUP BY 1 ORDER BY 1;
-- 7. Top 20 customers to contact first
SELECT CustomerID, Customer_Value, Churn_Probability, Value_At_Risk, Retention_Action
FROM churn ORDER BY Value_At_Risk DESC LIMIT 20;
-- 8. Subscription band churn
SELECT Subscription_Band AS grp, COUNT(*) AS n, ROUND(100.0*AVG(Churn),1) AS churn_pct FROM churn GROUP BY 1 ORDER BY 1;

-- ===== EXPLORATION: churn by the remaining guideline variables (quartile bins) =====
-- 9. Call failures
SELECT CASE WHEN Call_Failure=0 THEN '0' WHEN Call_Failure<=5 THEN '1-5' WHEN Call_Failure<=10 THEN '6-10' ELSE '11+' END AS grp,
       COUNT(*) AS n, ROUND(100.0*AVG(Churn),1) AS churn_pct FROM churn GROUP BY 1 ORDER BY MIN(Call_Failure);
-- 10. Frequency of use
SELECT CASE WHEN Frequency_of_use<=10 THEN '0-10' WHEN Frequency_of_use<=30 THEN '11-30' WHEN Frequency_of_use<=60 THEN '31-60' ELSE '61+' END AS grp,
       COUNT(*) AS n, ROUND(100.0*AVG(Churn),1) AS churn_pct FROM churn GROUP BY 1 ORDER BY MIN(Frequency_of_use);
-- 11. Frequency of SMS
SELECT CASE WHEN Frequency_of_SMS=0 THEN '0' WHEN Frequency_of_SMS<=10 THEN '1-10' WHEN Frequency_of_SMS<=50 THEN '11-50' ELSE '51+' END AS grp,
       COUNT(*) AS n, ROUND(100.0*AVG(Churn),1) AS churn_pct FROM churn GROUP BY 1 ORDER BY MIN(Frequency_of_SMS);
-- 12. Seconds of use
SELECT CASE WHEN Seconds_of_Use<=500 THEN '0-500' WHEN Seconds_of_Use<=2000 THEN '501-2000' WHEN Seconds_of_Use<=5000 THEN '2001-5000' ELSE '5001+' END AS grp,
       COUNT(*) AS n, ROUND(100.0*AVG(Churn),1) AS churn_pct FROM churn GROUP BY 1 ORDER BY MIN(Seconds_of_Use);
-- 13. Customer value
SELECT Value_Tier AS grp, COUNT(*) AS n, ROUND(100.0*AVG(Churn),1) AS churn_pct, ROUND(AVG(Customer_Value),0) AS avg_value FROM churn GROUP BY 1 ORDER BY avg_value;
-- ===== DATA QUALITY =====
-- 14. Duplicate check (run on the raw table, here: customer IDs must be unique)
SELECT COUNT(*) AS rows_total, COUNT(DISTINCT CustomerID) AS unique_ids FROM churn;
-- 15. Outlier check: top 1% of customers by usage
SELECT COUNT(*) AS n, ROUND(100.0*AVG(Churn),1) AS churn_pct, ROUND(AVG(Customer_Value),0) AS avg_value
FROM churn WHERE Seconds_of_Use > (SELECT Seconds_of_Use FROM churn ORDER BY Seconds_of_Use DESC LIMIT 1 OFFSET (SELECT COUNT(*)/100 FROM churn));
