# Data

| File | Description |
|---|---|
| `raw/Customer_Churn.csv` | Original Iranian Churn dataset (3,150 rows, 14 columns) |
| `processed/churn_clean.csv` | Cleaned data: 300 duplicates removed, 2,850 unique customers, customer ID, readable labels, 12 model inputs and the target |
| `processed/churn_scored.csv` | Cleaned data plus model output: churn probability, value at risk, risk band, value tier, retention action and priority rank |

## Source

Iranian Churn Dataset, UCI Machine Learning Repository (donated 2020).
https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset
DOI: 10.24432/C5JW3Z. Licensed under CC BY 4.0 (attribution required).

The data was collected from an Iranian telecom company. The inputs summarize the first 9 months of customer activity; the churn label is the state of the customer at the end of 12 months, which leaves a 3-month planning gap.
The records are anonymous customer-level data and contain no personal information.

## Data dictionary

| Variable | Meaning |
|---|---|
| Call_Failure | Number of failed calls |
| Complains | 1 if the customer filed a complaint |
| Subscription_Length | Months since the account was opened |
| Charge_Amount | Billing band (0 = lowest) |
| Seconds_of_Use | Total call seconds |
| Frequency_of_use | Total number of calls |
| Frequency_of_SMS | Total number of SMS |
| Distinct_Called_Numbers | Number of different numbers called |
| Tariff_Plan | 1 = Pay as you go, 2 = Contractual |
| Status | 1 = Active, 2 = Non-active |
| Age / Age_Group | Customer age and age band (Age_Group is dropped from the models) |
| Customer_Value | Calculated value of the customer (no currency given in the source) |
| Churn | Target: 1 = left, 0 = stayed |

### Columns added by the project (scored file)

`Churn_Probability`, `Value_At_Risk` (probability x customer value), `Expected_Retention_Benefit` (value at risk x 0.40 - 20), `Contact_Recommended`, `Risk_Band` (High >= 0.5, Medium 0.2-0.5, Low < 0.2), `Value_Tier` (Standard / Mid-Value / High-Value), `Retention_Action`, `Priority_Rank`.
