-- Feature layer: one row per transaction, model-ready columns.
--
-- Rule for every behavioural feature: it may only look at transactions that
-- happened BEFORE the current one. That is what a real-time scoring system
-- can know at the moment a payment arrives, and it prevents target leakage.

CREATE SCHEMA IF NOT EXISTS features;
DROP TABLE IF EXISTS features.transactions;

CREATE TABLE features.transactions AS
WITH base AS (
    SELECT
        t.transaction_id,
        t.is_fraud,
        t.transaction_dt,
        t.transaction_dt / 86400                         AS day,
        t.transaction_amt::float8                        AS amt,
        t.product_cd,
        t.card1, t.card2, t.card3, t.card4, t.card5, t.card6,
        t.addr1, t.addr2, t.dist1,
        t.p_emaildomain, t.r_emaildomain,
        t.c1, t.c2, t.c5, t.c6, t.c9, t.c11, t.c13, t.c14,
        t.d1, t.d2, t.d3, t.d4, t.d10, t.d15,
        t.m4, t.m5, t.m6,
        (i.transaction_id IS NOT NULL)::int              AS has_identity,
        i.device_type,
        -- Approximate customer ID. Vesta does not give a client ID, but D1 is
        -- "days since the card was first used", so (day - D1) is the card's
        -- start day and stays constant for one customer. card1 + addr1 +
        -- start day is a well-known proxy for "the same person".
        t.card1::text || '_' || COALESCE(t.addr1::text, 'na') || '_' ||
            COALESCE(((t.transaction_dt / 86400) - t.d1::int)::text, 'na') AS uid
    FROM raw.train_transaction t
    LEFT JOIN raw.train_identity i USING (transaction_id)
)
SELECT
    b.*,

    -- Time of day: fraud often happens at night, when the owner is asleep.
    ((transaction_dt / 3600) % 24)::int                  AS hour,

    -- Cents part of the amount. Odd cents (e.g. 49.817) usually mean a
    -- currency conversion, i.e. a foreign merchant.
    ROUND((amt - FLOOR(amt))::numeric, 3)::float8        AS amt_cents,
    LN(1 + amt)                                          AS log_amt,

    -- Payer and recipient email domains differ: typical for card testing.
    (p_emaildomain IS DISTINCT FROM r_emaildomain
        AND r_emaildomain IS NOT NULL)::int              AS email_mismatch,

    -- Customer history: how much has this "person" done before?
    COUNT(*) OVER w_uid_past                             AS uid_prev_tx,
    AVG(amt) OVER w_uid_past                             AS uid_prev_mean_amt,
    STDDEV_SAMP(amt) OVER w_uid_past                     AS uid_prev_std_amt,

    -- Velocity: many payments in a short window is the classic fraud signal
    -- (a stolen card is used fast, before it gets blocked).
    COUNT(*) OVER w_uid_1h                               AS uid_tx_last_1h,
    COUNT(*) OVER w_uid_24h                              AS uid_tx_last_24h,
    COUNT(*) OVER w_card_24h                             AS card1_tx_last_24h,

    -- Seconds since this customer's previous transaction.
    transaction_dt - LAG(transaction_dt) OVER w_uid_order AS secs_since_prev_tx,

    -- Has this customer paid this exact amount before? Repeated amounts are
    -- typical of subscriptions, i.e. legitimate behaviour.
    COUNT(*) OVER w_uid_amt_past                         AS uid_same_amt_prev
FROM base b
WINDOW
    w_uid_order    AS (PARTITION BY uid ORDER BY transaction_dt, transaction_id),
    w_uid_past     AS (PARTITION BY uid ORDER BY transaction_dt, transaction_id
                       ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
    w_uid_1h       AS (PARTITION BY uid ORDER BY transaction_dt
                       RANGE BETWEEN 3600 PRECEDING AND 1 PRECEDING),
    w_uid_24h      AS (PARTITION BY uid ORDER BY transaction_dt
                       RANGE BETWEEN 86400 PRECEDING AND 1 PRECEDING),
    w_card_24h     AS (PARTITION BY card1 ORDER BY transaction_dt
                       RANGE BETWEEN 86400 PRECEDING AND 1 PRECEDING),
    w_uid_amt_past AS (PARTITION BY uid, amt ORDER BY transaction_dt, transaction_id
                       ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING);

-- Ratio of this amount to the customer's usual amount: "is this payment
-- unusually large FOR THIS PERSON?" Raw amount alone barely separates fraud
-- (see 03_data_quality.sql), but relative amount does.
ALTER TABLE features.transactions ADD COLUMN amt_to_uid_mean float8;
UPDATE features.transactions
SET amt_to_uid_mean = amt / NULLIF(uid_prev_mean_amt, 0);

ALTER TABLE features.transactions ADD PRIMARY KEY (transaction_id);
ANALYZE features.transactions;

-- Sanity check: velocity features vs fraud rate.
SELECT LEAST(uid_tx_last_24h, 5)          AS tx_in_prev_24h_capped,
       COUNT(*)                           AS n,
       ROUND(100.0 * AVG(is_fraud), 2)    AS fraud_rate_pct
FROM features.transactions
GROUP BY 1
ORDER BY 1;
