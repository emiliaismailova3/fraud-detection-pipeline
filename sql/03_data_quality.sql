-- Data quality checks on the raw layer. Each query answers one question
-- that decides how we model the data later.

-- 1. How many rows, and how imbalanced is the target?
--    (Tells us accuracy is useless: predicting "never fraud" is ~96.5% accurate.)
SELECT COUNT(*)                          AS n_transactions,
       SUM(is_fraud)                     AS n_fraud,
       ROUND(100.0 * AVG(is_fraud), 2)   AS fraud_rate_pct
FROM raw.train_transaction;

-- 2. Any duplicate transaction IDs? (PRIMARY KEY already forbids them; this
--    is the check you would run if it didn't.)
SELECT COUNT(*) - COUNT(DISTINCT transaction_id) AS duplicate_ids
FROM raw.train_transaction;

-- 3. What share of transactions has identity (device/browser) data?
SELECT ROUND(100.0 * COUNT(i.transaction_id) / COUNT(*), 1) AS pct_with_identity,
       ROUND(100.0 * AVG(t.is_fraud) FILTER (WHERE i.transaction_id IS NOT NULL), 2)
           AS fraud_rate_with_identity_pct,
       ROUND(100.0 * AVG(t.is_fraud) FILTER (WHERE i.transaction_id IS NULL), 2)
           AS fraud_rate_without_identity_pct
FROM raw.train_transaction t
LEFT JOIN raw.train_identity i USING (transaction_id);

-- 4. Time span. transaction_dt is seconds from an unknown reference point,
--    so we express it in days. A time-ordered split needs this.
SELECT ROUND(MIN(transaction_dt) / 86400.0, 1) AS first_day,
       ROUND(MAX(transaction_dt) / 86400.0, 1) AS last_day
FROM raw.train_transaction;

-- 5. Fraud rate per week: is the rate stable over time?
SELECT (transaction_dt / (7 * 86400))   AS week,
       COUNT(*)                         AS n,
       ROUND(100.0 * AVG(is_fraud), 2)  AS fraud_rate_pct
FROM raw.train_transaction
GROUP BY week
ORDER BY week;

-- 6. Missing values in the columns we plan to use.
SELECT ROUND(100.0 * AVG((card2 IS NULL)::int), 1)          AS card2_null_pct,
       ROUND(100.0 * AVG((addr1 IS NULL)::int), 1)          AS addr1_null_pct,
       ROUND(100.0 * AVG((dist1 IS NULL)::int), 1)          AS dist1_null_pct,
       ROUND(100.0 * AVG((p_emaildomain IS NULL)::int), 1)  AS p_email_null_pct,
       ROUND(100.0 * AVG((r_emaildomain IS NULL)::int), 1)  AS r_email_null_pct,
       ROUND(100.0 * AVG((d1 IS NULL)::int), 1)             AS d1_null_pct
FROM raw.train_transaction;

-- 7. Fraud rate by product and card type: are categorical columns informative?
SELECT product_cd, card6,
       COUNT(*)                         AS n,
       ROUND(100.0 * AVG(is_fraud), 2)  AS fraud_rate_pct
FROM raw.train_transaction
GROUP BY product_cd, card6
HAVING COUNT(*) > 1000
ORDER BY fraud_rate_pct DESC;

-- 8. Amount distribution, fraud vs legit.
SELECT is_fraud,
       ROUND(AVG(transaction_amt), 2)                                          AS mean_amt,
       ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY transaction_amt)::numeric, 2) AS median_amt,
       ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY transaction_amt)::numeric, 2) AS p99_amt
FROM raw.train_transaction
GROUP BY is_fraud;
