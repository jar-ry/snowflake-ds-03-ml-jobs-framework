-- =============================================================================
-- CHURN CLASSIFICATION DEMO — Teardown Script
-- =============================================================================
-- Drops all objects created by setup.sql in reverse dependency order.
-- Safe to run multiple times (uses IF EXISTS).
-- =============================================================================

USE ROLE ACCOUNTADMIN;

-- ─── Drop database (cascades all schemas, tables, views, stages, etc.) ───────
DROP DATABASE IF EXISTS CHURN_CLASSIFICATION_DEMO;

-- ─── Drop warehouse ──────────────────────────────────────────────────────────
DROP WAREHOUSE IF EXISTS CHURN_CLASSIFICATION_DEMO_WH;

-- ─── Drop role ───────────────────────────────────────────────────────────────
DROP ROLE IF EXISTS CHURN_CLASSIFICATION_DEMO_ROLE;
