# Production Design (MVP-focused): Forecasting Workflow for T+1..T+6

**Author:** Ignacio Layo González  
**Context:** Technical design proposal for deploying the forecasting pipeline (Exercise 1) into a realistic, maintainable, and scalable production setup.

---

## Executive summary

This document outlines how the forecasting pipeline developed in Exercise 1 could evolve into a production-ready system.  
The focus is on a **simple, pragmatic stack** that a single engineer can implement and maintain, while also showing how it could grow into a more advanced architecture if the project scales.

---

## MVP (Single-engineer, defendable stack)

**Goal:** Reproducible pipeline that produces T+1..T+6 hourly forecasts from the last ESIOS datapoint (T).

### Core components
- **Ingestion:** `run_pipeline.py` pulls ESIOS API data (HTTP request) and stores raw parquet files in `data/raw/`.  
- **Feature engineering:** `scripts/process_features.py` converts 5-min readings to hourly averages, adds lag and rolling features.  
- **Model training & prediction:** `src/model.py` (scikit-learn RandomForest) trains on past data and recursively predicts the next 6 hours.  
- **Prediction storage:** parquet files in `data/predictions/` (or a lightweight Postgres DB for serving).  
- **Serving:** small REST API using **FastAPI** that exposes the latest predictions for the trading application.  
- **CI / reproducibility:** GitHub repository with unit tests, optionally integrated with GitHub Actions.  
- **Deployment:** packaged as a Docker image, runnable locally or on a simple VM / Azure App Service.

### Why this stack
- Fully reproducible and realistic for a single engineer.  
- Pure Python; no dependency on heavy cloud services.  
- Easy to explain and maintain.  
- Can later integrate naturally with standard cloud tools if required.

---

## Simplified architecture diagram

```mermaid
flowchart LR
  A[ESIOS API] -->|HTTP| B[run_pipeline.py]
  B --> C[data/raw/*.parquet]
  C --> D[scripts/process_features.py]
  D --> E[data/processed/features_*.parquet]
  E --> F[src/model.py (train + predict)]
  F --> G[data/predictions/predictions_*.parquet]
  G --> H[FastAPI endpoint] --> I[Trading App]
```

Component details & responsibilities
1. Ingestion & storage

Azure Data Factory (ADF) / Azure Functions: scheduled pulls from ESIOS (HTTP) and backups; small functions handle token rotation and retries.

Event Hubs (or Kafka): ingest streaming sources (telemetry, market changes) for lower-latency updates.

Azure Data Lake Storage (ADLS Gen2) + Delta Lake: raw immutable storage for traceability. Use partitioning by date and indicator ID.

2. Processing, features & storage

Azure Databricks (Spark + Delta):

Batch jobs: resample raw 5-minute data to hourly, imputations, generate lags & rolling stats, enrich with calendar features.

Job outputs: materialized hourly view and features Delta tables (time-partitioned).

Feature Store:

Use Delta as canonical store and optionally integrate Feast or Databricks Feature Store for online lookups.

Persist feature metadata and freshness timestamps.

3. Training & model management

Databricks / AzureML training jobs:

Parameterized notebooks (or job tasks) that read features from Delta, train model, evaluate, log metrics.

Persist model artifacts and metrics to MLflow (model registry) and store artifacts in blob storage.

Model registry / governance:

MLflow or Azure ML Model Registry to track versions, stage (staging/production), and lineage (dataset version, hyperparameters).

Automated model tests (smoke tests, performance gates).

4. Serving predictions

Online serving (low-latency):

Deploy model as a REST endpoint on AKS using KServe / KFServing or Azure ML real-time endpoints.

Alternatively, use Azure Functions for lightweight models, but prefer AKS for autoscaling and more complex inference.

Batch predictions & storage:

Orchestrated scheduled predictions (e.g., every time new data at T arrives).

Write predictions to:

Delta table for historical retention,

Azure SQL / Cosmos (for trading app queries),

or push via Service Bus / Event Grid to the trading system.

API:

Azure API Management in front of model endpoints: enforce authentication (Azure AD), rate limiting, discovery for trading app.

5. Consumer integration (Trading app)

Two integration patterns:

Pull: Trading app calls REST endpoint (API Management) to fetch predictions for desired horizon.

Push: Pipeline publishes predictions to Service Bus / Event Grid or writes to DB; trading app subscribes for updates.

Prefer push for deterministic delivery or database integration for historical queries.

6. Orchestration & automation

Azure Data Factory / Azure Pipelines / GitHub Actions:

Orchestrate ETL → features → train → evaluate → register → serve.

Retrain triggered by schedule, by drift detection event, or by manual approval.

Infrastructure as Code: Terraform or Bicep for reproducible infrastructure.

7. Monitoring, observability & drift detection

Application Insights / Log Analytics: monitor pipeline logs, latency, failures.

Prometheus + Grafana (or Azure Monitor + PowerBI): visualize metrics — throughput, latency, model response time, error rates.

Model health:

Track prediction and feature distributions.

Implement data-drift and concept-drift detectors (e.g., KS test vs. training baseline).

Automatic alerting (PagerDuty / Teams / Slack) when drift crosses thresholds.

Evaluation: store ground-truth when available; compute backfilled MAE/RMSE and attach to model runs.

8. Security & compliance

Key Vault: manage secrets (API keys, DB credentials).

Managed Identities: for services to avoid plain secrets in code.

Network isolation: private endpoints for storage and compute; RBAC for resource access.

Audit & lineage: tracked via MLflow and Purview / Data Catalog.

Dataflow (stepwise)

ETL: Scheduled Data Factory job pulls ESIOS → raw Delta in ADLS.

Aggregate: Databricks job resamples to hourly view and writes features table.

Train: Triggered by schedule or drift; reads features, trains model, logs to MLflow.

Register: If metrics pass, model promoted to staging/production in MLflow registry.

Deploy: CI pipeline (GitHub Actions) builds container, pushes to ACR, deploys to AKS endpoint.

Predict: On new data arrival, model produces T+1..T+6 forecasts → DB + message event.

Consume: Trading app reads DB or subscribes to events; API Management enforces auth.

Non-functional requirements & design trade-offs

Latency: predictions available within minutes after T (batch hourly + AKS serving).

Reliability: Delta Lake + retry logic in Data Factory + AKS autoscaling ensure robustness.

Reproducibility: dataset and environment versions logged in MLflow and infra code.

Cost: optimize Databricks clusters and AKS instances; use spot instances for training jobs.

Operational runbook (summary)

Deploy: CI pipeline provisions infrastructure (Terraform) → deploys model container.

Monitor: dashboards display metrics; alerts for job or drift anomalies.

Rollback: MLflow registry enables reverting to previous model versions.

Incident response: if service down, trading app falls back to last stored predictions.

Roadmap / priority list (for delivering production)
Essential (MVP → Production-ready)

Automated ETL to ADLS + hourly feature materialization.

MLflow tracking + model registry.

Containerized model endpoint with API Management and authentication.

Prediction storage and a simple DB or message queue for the trading app.

Important (observability & reliability)

Monitoring dashboards, alerting, data drift detection.

Infrastructure as Code (Terraform / Bicep).

Nice-to-have (scale & governance)

Feature Store for online lookups (Feast / Databricks FS).

Multi-output direct forecasting models; online learning.

MLflow/Databricks integration for scheduled retraining and automatic promotion.

Appendix: Minimal infra pieces to prototype quickly

ADLS Gen2 storage + Delta tables.

Databricks (small job cluster) for feature generation.

GitHub Actions for CI: run unit tests and build model container.

AKS with Horizontal Pod Autoscaler for serving.

API Management + Azure AD for secure access.