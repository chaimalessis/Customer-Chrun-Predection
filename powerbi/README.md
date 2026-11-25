# Power BI dashboard

`Customer_Churn_Dashboard.pbix` is the interactive dashboard (open it with Power BI Desktop).
`dashboard_preview.pdf` is an export of the four pages, so it can be viewed without Power BI.

| Page | Content |
|---|---|
| 1. Executive Overview | KPI cards and churn rate by tariff plan, complaints, subscription length and status |
| 2. Customer Risk | Customers by risk category, churn probability vs customer value, top customers |
| 3. Retention Plan | Business assumptions, expected retention benefit, risk x value matrix, value at risk by action |
| 4. Model | Model comparison, feature importance, threshold cost curve, calibration curve |

Every page has filters for risk band, tariff plan and status. Data source: `data/processed/churn_scored.csv` and the CSV files in `results/`.
