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

### 1. An IAM user to deploy as

There is **no AWS managed policy for Lightsail access.** Searching the managed
policies for "lightsail" returns only `LightsailExportAccess`, which is for
exporting snapshots to EC2 and is not this. So the permissions are a
customer-managed policy, and [`iam-policy.json`](iam-policy.json) is one
scoped to exactly what `deploy.sh` calls — create and update the container
service, log in to its registry, register an image, deploy, read logs, and
delete the service when you are done.

In the console: **IAM → Policies → Create policy → JSON**, paste that file,
name it `PolicygateLightsailDeploy`. Then **IAM → Users → Create user**, attach
that policy, and under **Security credentials → Create access key** choose
*Command Line Interface (CLI)*.

Creating the policy and attaching it are two separate acts, and the console
lets you finish the first without the second. Check before deploying:

```bash
aws sts get-caller-identity                    # which user you are
aws lightsail get-container-services           # should list nothing, not deny
```

An `AccessDeniedException` naming `lightsail:GetContainerServices` means the
policy exists but is not attached to this user. The deploy user cannot check
its own permissions with `iam:ListAttachedUserPolicies` — that is deliberate,
so the console is where you look: **IAM → Users →** your user **→
Permissions**.

Use an IAM user, not the root account: root cannot be scoped, cannot be
rotated without changing your sign-in, and these keys end up in a shell
history.

### 2. The CLI and the Lightsail plugin

`push-container-image` is not pure CLI — it shells out to `lightsailctl`, and
without the plugin the push step fails:

```bash
brew install awscli
curl -L "https://s3.us-west-2.amazonaws.com/lightsailctl/latest/darwin-arm64/lightsailctl" \
  -o /opt/homebrew/bin/lightsailctl && chmod +x /opt/homebrew/bin/lightsailctl
```

(`darwin-arm64` for Apple Silicon; `darwin-amd64` for Intel.)

AWS's own instructions say `/usr/local/bin`, which on a default Apple Silicon
Mac is **not writable without sudo** — the curl fails and the only sign is an
error scrolling past, so the next thing to fail is the image push, twenty
minutes into a build. `/opt/homebrew/bin` is Homebrew's own directory, already
on PATH and already writable. Check it landed before moving on:

```bash
which lightsailctl
```

There is no Homebrew formula for it; the download is the only route.

### 3. Credentials and a region

```bash
aws configure          # the access key from step 1
```

The region must match the one `deploy.sh` uses — it defaults to
`ap-southeast-2` (Sydney); `ap-southeast-1` is Singapore. Pass `AWS_REGION` to
override it.

### 4. A budget alert, before the first deployment

A container service bills hourly for as long as it exists, whether or not
anyone visits it. **Billing → Budgets → Create budget**, a small monthly
figure with an email alert, is what stops a demo you stopped showing from
quietly costing $40 a month. Deleting the service is the other half; see the
bottom of this file.

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
| new-account credit | AWS replaced Lightsail's old container trial with the Free Tier's credits (up to $200); check Billing → Credits |
| data transfer | 500 GB included |
| model calls | whatever visitors spend, bounded per session by `POLICYGATE_SESSION_USD_CAP` |

A session's spend is shown on the Live session page as it accrues, and the
service refuses a turn that would start past the cap -- checked before the call,
not after.

Tear it down when you stop showing it; it bills while it exists:

```bash
aws lightsail delete-container-service --service-name policygate --region ap-southeast-2
```
