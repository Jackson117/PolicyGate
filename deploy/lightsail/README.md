# Deploying to Lightsail

One container service, two containers, one HTTPS URL.

```
                    https://policygate.<id>.ap-southeast-2.cs.amazonlightsail.com
                                        |
                              +---------v---------+
                              |     dashboard     |  public endpoint
                              |  Streamlit :8501  |  password-gated
                              +---------+---------+
                                        | localhost:8000
                              +---------v---------+
                              |      service      |  no public port
                              | FastAPI/LangGraph |  per-session spend cap
                              +-------------------+
```

Containers in one Lightsail deployment reach each other over `localhost`, and
only one container can be the public endpoint. So the approval service has no
route in from the internet: the only way to make it call a model is through the
dashboard, and the dashboard asks for a password first.

## Once

Install the CLI and the Lightsail plugin, then sign in:

```bash
brew install awscli
curl -L "https://s3.us-west-2.amazonaws.com/lightsailctl/latest/darwin-arm64/lightsailctl" \
  -o /usr/local/bin/lightsailctl && chmod +x /usr/local/bin/lightsailctl
aws configure          # an IAM user with Lightsail access; region ap-southeast-2 for Sydney
```

## Every time

```bash
set -a; . ./.env; set +a                       # OPENAI_API_KEY
export POLICYGATE_DASHBOARD_PASSWORD=...       # what you hand out with the link
./deploy/lightsail/deploy.sh
```

It stages the run data, builds both images, pushes them, deploys, waits, and
prints the URL. Ten to twenty minutes the first time, mostly uploading two
images of about 2 GB each.

## Sizing

Measured in the containers, not estimated:

| | resident |
|---|---|
| service, idle | 262 MB |
| dashboard, `w4_v4` loaded | 604 MB |
| dashboard, `w4_v4_ablation` loaded | 1.07 GB |

The ablation run is 170 MB of JSON that parses to about a gigabyte of objects,
so **`medium` (2 GB, $40/month) is what holds both runs.** `micro` (1 GB,
$10/month) fits if you ship one run:

```bash
LIGHTSAIL_POWER=micro LIGHTSAIL_RUNS=w4_v4 ./deploy/lightsail/deploy.sh
```

The Ablation page then has only the headline arms to show, because the
single-mechanism arms live in the run that was left out.

`POLICYGATE_RUN_CACHE` bounds how many runs are held at once. It defaults to 1
here: the cache is per server process and shared by everyone connected, so
without a bound a visitor clicking through the run selector loads every run the
image ships until the container dies.

## What this deployment deliberately does not have

- **A database.** A container service has no volume, so the audit store is
  SQLite inside the container and the LangGraph checkpointer is in memory. Both
  are lost on redeploy, and a pending approval does not survive a restart. For
  anything beyond a demo, point `POLICYGATE_AUDIT_URL` and
  `POLICYGATE_CHECKPOINT_URL` at a managed database.
- **Accounts.** One shared password, no identity, nothing logged about who
  looked. Enough to keep a crawler out of a demo; not an access control system.
- **The Retrieval page.** It needs the policygate-rag checkout itself -- the
  corpus and the embedding cache are resolved relative to that repository's
  root -- so in the container the page says it is unavailable. The Banking RAG
  page, which reads that study's published `results/`, does ship and works.

## Costs

| | |
|---|---|
| container service, `medium` | $40/month |
| container service, `micro` | $10/month |
| data transfer | 500 GB included |
| model calls | whatever visitors spend, bounded per session by `POLICYGATE_SESSION_USD_CAP` |

A session's spend is shown on the Live session page as it accrues, and the
service refuses a turn that would start past the cap -- checked before the call,
not after.

Tear it down when you stop showing it; it bills while it exists:

```bash
aws lightsail delete-container-service --service-name policygate --region ap-southeast-2
```
