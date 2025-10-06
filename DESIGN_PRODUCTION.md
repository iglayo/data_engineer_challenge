# Production Design: Forecasting Workflow for T+1..T+6 Demand (Technical Design)

**Author:** Ignacio Layo González  
**Context:** Builds on the Exercise 1 MVP (ETL → hourly features → RandomForest → recursive T+1..T+6 forecasts).  
**Target:** Production-grade, automated, secure, observable pipeline suitable for integration into a trading team’s systems.

---

## Executive summary (short)
This document describes a cloud architecture and operational design to convert the local MVP into a production service that:
- Ingests streaming and batch market data (ESIOS, telemetry),
- Produces hourly forecasts for the next 6 hours from the last observed data point (T),
- Serves predictions to downstream trading applications via secure, low-latency interfaces,
- Enables automated retraining, model governance, monitoring (data & model drift), and reproducible deployments.

Platform choice: **Microsoft Azure** (matches team competence: Databricks / Azure ML / Event Hubs / AKS). The design is modular and can be adapted to AWS/GCP.

---

## High-level architecture

```mermaid
flowchart LR
  subgraph Ingest & Storage
    A[ESIOS API / external feeds] -->|HTTP / Pull| B[Azure Data Factory / Functions]
    A --> |Event Stream| C[Event Hubs / Kafka]
    B --> D[Raw storage (Azure Data Lake Storage / Delta Lake)]
    C --> D
  end

  subgraph Processing & Feature Store
    D --> E[Azure Databricks (Spark) Delta Lake]
    E --> F[Feature engineering & materialize hourly view]
    F --> G[Feature Store (Delta + MLflow/Feast)]
    G --> H[Training jobs (Databricks or AzureML)]
  end

  subgraph Model Registry & Serving
    H --> I[MLflow Model Registry / Azure ML Model Registry]
    I --> J[Model CI/CD (GitHub Actions / Azure DevOps)]
    J --> K[Model Serving: AKS + KFServing / Azure ML Endpoint]
  end

  subgraph Orchestration & Automation
    L[Azure Data Factory / Azure Pipelines] --> E
    L --> H
    L --> M[Trigger: retrain on drift or schedule]
  end

  subgraph Serving & Consumers
    K --> N[API Gateway / Azure API Management]
    N --> O[Trading App (internal) via secured REST]
    K --> P[Predictions storage (Azure SQL / Cosmos / Delta)]
    P --> O
    K --> Q[Message queue: Azure Service Bus / Event Grid]
    Q --> O
  end

  subgraph Observability & Governance
    Datasets[(Delta + Parquet versions)] ---|catalog| R[Data Catalog / Purview]
    All --> S[Application Insights / Log Analytics]
    S --> T[Grafana / PowerBI dashboards]
    I --> U[Model lineage & audit (MLflow)]
    H --> V[Model evaluation reports (Artifact store)]
  end
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