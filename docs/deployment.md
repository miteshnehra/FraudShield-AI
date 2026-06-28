# Deployment Guide

## Local Windows Deployment

Open PowerShell in the project folder:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pytest
streamlit run app.py
```

If PowerShell blocks activation, use Command Prompt:

```bat
py -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements-dev.txt
pytest
streamlit run app.py
```

Open `http://localhost:8501`.

## Docker

Build and start:

```bash
docker compose up --build
```

Stop:

```bash
docker compose down
```

The compose file persists model bundles, reports, and logs in local project folders. Uploaded datasets remain in Streamlit session memory unless a user explicitly downloads or saves an artifact.

## Streamlit Community Cloud

1. Push the project to a GitHub repository.
2. Sign in to Streamlit Community Cloud.
3. Create an app from the repository.
4. Set the entry point to `app.py`.
5. Deploy and verify `/_stcore/health`.

The free tier has resource limits. SHAP and three-model cross-validation can be slow or memory-intensive. Reduce candidate models or disable cross-validation for a demo deployment.

## Render

The included `render.yaml` is a blueprint for a free demo web service:

1. Push the repository to GitHub.
2. In Render, create a new Blueprint.
3. Select the repository.
4. Review the service and deploy.
5. Confirm the health check passes.

Free instances can sleep, restart, or lose local filesystem artifacts. Download model and report files instead of treating free-instance disk as durable storage.

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `FRAUDSHIELD_ENV` | `development` | Runtime label |
| `FRAUDSHIELD_LOG_LEVEL` | `INFO` | Application log level |

No API key is required in v1.0.

## Public Deployment Warning

The portfolio application has no authentication, tenant isolation, encryption key management, durable audit store, or formal retention controls. Do not upload real payment, identity, customer, or regulated financial data to a public demo.

Production deployment requires:

- Authentication and role-based authorization
- TLS and encryption at rest
- Secret management
- Durable audit logging
- Data retention and deletion policies
- Privacy, legal, and fairness review
- Monitoring, alerting, and incident response
- Model registry and approval workflow
- Human-review case management

