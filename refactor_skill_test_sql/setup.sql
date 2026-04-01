-- =============================================================================
-- CHURN CLASSIFICATION DEMO — Setup Script
-- =============================================================================
-- Creates:
--   Database:  CHURN_CLASSIFICATION_DEMO
--   Schemas:   DS, FEATURE_STORE, MODELLING
--   Role:      CHURN_CLASSIFICATION_DEMO_ROLE
--   Warehouse: CHURN_CLASSIFICATION_DEMO_WH
--   Tables:    DS.SUBSCRIBERS, DS.USAGE_BEHAVIOR
--   Data:      ~2000 synthetic subscriber rows with realistic churn patterns
-- =============================================================================

USE ROLE ACCOUNTADMIN;

-- ─── Role ────────────────────────────────────────────────────────────────────
CREATE ROLE IF NOT EXISTS CHURN_CLASSIFICATION_DEMO_ROLE;
GRANT ROLE CHURN_CLASSIFICATION_DEMO_ROLE TO ROLE SYSADMIN;

-- ─── Warehouse ───────────────────────────────────────────────────────────────
CREATE WAREHOUSE IF NOT EXISTS CHURN_CLASSIFICATION_DEMO_WH
    WAREHOUSE_SIZE = 'XSMALL'
    AUTO_SUSPEND   = 60
    AUTO_RESUME    = TRUE;

GRANT USAGE ON WAREHOUSE CHURN_CLASSIFICATION_DEMO_WH
    TO ROLE CHURN_CLASSIFICATION_DEMO_ROLE;

-- ─── Database & Schemas ──────────────────────────────────────────────────────
CREATE DATABASE IF NOT EXISTS CHURN_CLASSIFICATION_DEMO;

GRANT OWNERSHIP ON DATABASE CHURN_CLASSIFICATION_DEMO
    TO ROLE CHURN_CLASSIFICATION_DEMO_ROLE
    COPY CURRENT GRANTS;

USE ROLE CHURN_CLASSIFICATION_DEMO_ROLE;
USE DATABASE CHURN_CLASSIFICATION_DEMO;
USE WAREHOUSE CHURN_CLASSIFICATION_DEMO_WH;

CREATE SCHEMA IF NOT EXISTS DS;
CREATE SCHEMA IF NOT EXISTS FEATURE_STORE;
CREATE SCHEMA IF NOT EXISTS MODELLING;

-- ─── SUBSCRIBERS table ───────────────────────────────────────────────────────
CREATE OR REPLACE TABLE DS.SUBSCRIBERS (
    SUBSCRIBER_ID      INT,
    AGE                INT,
    GENDER             VARCHAR(10),
    REGION             VARCHAR(20),
    ANNUAL_INCOME      FLOAT,
    PLAN_TYPE          VARCHAR(20),    -- 'basic', 'standard', 'premium'
    TENURE_MONTHS      INT,
    SIGNUP_DATE        DATE,
    CONTRACT_TYPE      VARCHAR(20),    -- 'month-to-month', 'one-year', 'two-year'
    PAYMENT_METHOD     VARCHAR(30),    -- 'credit_card', 'bank_transfer', 'digital_wallet'
    MONTHLY_CHARGE     FLOAT,
    UPDATED_AT         TIMESTAMP_NTZ
);

-- ─── USAGE_BEHAVIOR table ────────────────────────────────────────────────────
CREATE OR REPLACE TABLE DS.USAGE_BEHAVIOR (
    SUBSCRIBER_ID          INT,
    AVG_MONTHLY_SESSIONS   FLOAT,
    AVG_SESSION_DURATION   FLOAT,     -- minutes
    SUPPORT_TICKETS        INT,
    LAST_LOGIN_DATE        DATE,
    MONTHLY_DATA_USAGE_GB  FLOAT,
    OVERAGE_CHARGES        FLOAT,
    NUM_REFERRALS          INT,
    SATISFACTION_SCORE     INT,       -- 1-10
    CHURNED                INT,       -- 0 or 1 (target)
    UPDATED_AT             TIMESTAMP_NTZ
);

-- ─── Generate synthetic data ─────────────────────────────────────────────────
-- Uses a CTE with GENERATOR to create 2000 subscribers with realistic
-- correlations: higher churn for month-to-month, low satisfaction, low usage.

INSERT INTO DS.SUBSCRIBERS
WITH base AS (
    SELECT
        ROW_NUMBER() OVER (ORDER BY SEQ4()) AS SUBSCRIBER_ID,
        18 + UNIFORM(0, 52, RANDOM())       AS AGE,
        CASE UNIFORM(1, 3, RANDOM())
            WHEN 1 THEN 'male'
            WHEN 2 THEN 'female'
            ELSE        'other'
        END AS GENDER,
        CASE UNIFORM(1, 5, RANDOM())
            WHEN 1 THEN 'northeast'
            WHEN 2 THEN 'southeast'
            WHEN 3 THEN 'midwest'
            WHEN 4 THEN 'west'
            ELSE        'southwest'
        END AS REGION,
        ROUND(25000 + UNIFORM(0, 125000, RANDOM()), 2) AS ANNUAL_INCOME,
        CASE UNIFORM(1, 3, RANDOM())
            WHEN 1 THEN 'basic'
            WHEN 2 THEN 'standard'
            ELSE        'premium'
        END AS PLAN_TYPE,
        UNIFORM(1, 72, RANDOM())            AS TENURE_MONTHS,
        DATEADD('month',
                -UNIFORM(1, 72, RANDOM()),
                CURRENT_DATE())             AS SIGNUP_DATE,
        CASE UNIFORM(1, 3, RANDOM())
            WHEN 1 THEN 'month-to-month'
            WHEN 2 THEN 'one-year'
            ELSE        'two-year'
        END AS CONTRACT_TYPE,
        CASE UNIFORM(1, 3, RANDOM())
            WHEN 1 THEN 'credit_card'
            WHEN 2 THEN 'bank_transfer'
            ELSE        'digital_wallet'
        END AS PAYMENT_METHOD,
        ROUND(
            CASE
                WHEN PLAN_TYPE = 'basic'    THEN 19.99 + UNIFORM(0, 10, RANDOM())
                WHEN PLAN_TYPE = 'standard' THEN 39.99 + UNIFORM(0, 20, RANDOM())
                ELSE                             69.99 + UNIFORM(0, 30, RANDOM())
            END, 2
        ) AS MONTHLY_CHARGE,
        CURRENT_TIMESTAMP() AS UPDATED_AT
    FROM TABLE(GENERATOR(ROWCOUNT => 2000))
)
SELECT * FROM base;

INSERT INTO DS.USAGE_BEHAVIOR
WITH subscriber_base AS (
    SELECT
        SUBSCRIBER_ID,
        PLAN_TYPE,
        CONTRACT_TYPE,
        TENURE_MONTHS,
        MONTHLY_CHARGE
    FROM DS.SUBSCRIBERS
),
behavior AS (
    SELECT
        s.SUBSCRIBER_ID,
        -- Correlated usage: premium users have more sessions
        ROUND(
            CASE
                WHEN s.PLAN_TYPE = 'premium'  THEN 15 + UNIFORM(0, 20, RANDOM())
                WHEN s.PLAN_TYPE = 'standard' THEN 8 + UNIFORM(0, 15, RANDOM())
                ELSE                               2 + UNIFORM(0, 10, RANDOM())
            END, 1
        ) AS AVG_MONTHLY_SESSIONS,
        -- Session duration in minutes
        ROUND(5 + UNIFORM(0, 55, RANDOM()), 1) AS AVG_SESSION_DURATION,
        -- Support tickets: more for month-to-month
        CASE
            WHEN s.CONTRACT_TYPE = 'month-to-month' THEN UNIFORM(0, 8, RANDOM())
            ELSE                                          UNIFORM(0, 3, RANDOM())
        END AS SUPPORT_TICKETS,
        -- Last login: recent for active users, stale for churners
        DATEADD('day',
                -UNIFORM(0, 90, RANDOM()),
                CURRENT_DATE()) AS LAST_LOGIN_DATE,
        ROUND(
            CASE
                WHEN s.PLAN_TYPE = 'premium'  THEN 20 + UNIFORM(0, 80, RANDOM())
                WHEN s.PLAN_TYPE = 'standard' THEN 10 + UNIFORM(0, 40, RANDOM())
                ELSE                               1 + UNIFORM(0, 15, RANDOM())
            END, 2
        ) AS MONTHLY_DATA_USAGE_GB,
        ROUND(UNIFORM(0, 50, RANDOM()) * 0.5, 2) AS OVERAGE_CHARGES,
        UNIFORM(0, 5, RANDOM()) AS NUM_REFERRALS,
        UNIFORM(1, 10, RANDOM()) AS SATISFACTION_SCORE,
        -- Churn label: correlated with contract type, satisfaction, tenure
        -- Higher churn probability for: month-to-month, low satisfaction,
        -- low tenure, high support tickets
        0 AS CHURNED_PLACEHOLDER,
        CURRENT_TIMESTAMP() AS UPDATED_AT
    FROM subscriber_base s
)
SELECT
    SUBSCRIBER_ID,
    AVG_MONTHLY_SESSIONS,
    AVG_SESSION_DURATION,
    SUPPORT_TICKETS,
    LAST_LOGIN_DATE,
    MONTHLY_DATA_USAGE_GB,
    OVERAGE_CHARGES,
    NUM_REFERRALS,
    SATISFACTION_SCORE,
    -- Realistic churn logic: ~25-30% churn rate overall
    CASE
        WHEN (
            -- Base churn score from multiple factors
            (CASE WHEN b.SATISFACTION_SCORE <= 3 THEN 40 ELSE 0 END)
            + (CASE WHEN b.SUPPORT_TICKETS >= 5 THEN 25 ELSE 0 END)
            + (CASE WHEN b.AVG_MONTHLY_SESSIONS < 5 THEN 20 ELSE 0 END)
            + (CASE WHEN s.CONTRACT_TYPE = 'month-to-month' THEN 30 ELSE 0 END)
            + (CASE WHEN s.TENURE_MONTHS < 6 THEN 15 ELSE 0 END)
            + (CASE WHEN s.PLAN_TYPE = 'basic' THEN 10 ELSE 0 END)
            + UNIFORM(0, 30, RANDOM())  -- noise
        ) > 70 THEN 1
        ELSE 0
    END AS CHURNED,
    UPDATED_AT
FROM behavior b
JOIN DS.SUBSCRIBERS s USING (SUBSCRIBER_ID);

-- ─── Grant privileges ────────────────────────────────────────────────────────
GRANT ALL ON SCHEMA DS             TO ROLE CHURN_CLASSIFICATION_DEMO_ROLE;
GRANT ALL ON SCHEMA FEATURE_STORE  TO ROLE CHURN_CLASSIFICATION_DEMO_ROLE;
GRANT ALL ON SCHEMA MODELLING      TO ROLE CHURN_CLASSIFICATION_DEMO_ROLE;

GRANT SELECT ON ALL TABLES IN SCHEMA DS TO ROLE CHURN_CLASSIFICATION_DEMO_ROLE;

-- ─── Verify ──────────────────────────────────────────────────────────────────
SELECT 'SUBSCRIBERS'    AS TABLE_NAME, COUNT(*) AS ROW_COUNT FROM DS.SUBSCRIBERS
UNION ALL
SELECT 'USAGE_BEHAVIOR' AS TABLE_NAME, COUNT(*) AS ROW_COUNT FROM DS.USAGE_BEHAVIOR;

SELECT ROUND(AVG(CHURNED) * 100, 1) AS CHURN_RATE_PCT FROM DS.USAGE_BEHAVIOR;
