#!/usr/bin/env bash
# Build both images, push them to a Lightsail container service, and deploy.
#
#   ./deploy/lightsail/deploy.sh
#
# Needs the AWS CLI with the Lightsail plugin, credentials that can manage
# Lightsail, and a .env holding OPENAI_API_KEY. Everything else has a default.
#
# What it makes: one container service running two containers. The dashboard is
# the public endpoint; the service is reachable only from inside it, over
# localhost, which is how Lightsail links containers in one deployment. So the
# only way to reach the model is through the dashboard's password.
set -euo pipefail

cd "$(dirname "$0")/../.."

SERVICE=${LIGHTSAIL_SERVICE:-policygate}
REGION=${AWS_REGION:-ap-southeast-2}
POWER=${LIGHTSAIL_POWER:-medium}
SCALE=${LIGHTSAIL_SCALE:-1}
RUNS=${LIGHTSAIL_RUNS:-"w4_v4 w4_v4_ablation"}
CAP=${POLICYGATE_SESSION_USD_CAP:-0.25}
RUN_CACHE=${POLICYGATE_RUN_CACHE:-1}

: "${POLICYGATE_DASHBOARD_PASSWORD:?set POLICYGATE_DASHBOARD_PASSWORD -- the dashboard is public}"
: "${OPENAI_API_KEY:?set OPENAI_API_KEY (source .env first)}"
OPENAI_API_BASE=${OPENAI_API_BASE:-https://api.openai.com/v1}

aws() { command aws --region "$REGION" "$@"; }

echo "==> staging run data"
# shellcheck disable=SC2086
./scripts/stage_deploy.sh $RUNS

echo "==> building images"
docker build --target service   -t policygate-service:deploy   .
docker build --target dashboard -t policygate-dashboard:deploy .

if ! aws lightsail get-container-services --service-name "$SERVICE" >/dev/null 2>&1; then
  echo "==> creating container service $SERVICE ($POWER x$SCALE in $REGION)"
  aws lightsail create-container-service \
    --service-name "$SERVICE" --power "$POWER" --scale "$SCALE" >/dev/null
  echo "    waiting for it to be READY (a few minutes)"
  until [ "$(aws lightsail get-container-services --service-name "$SERVICE" \
              --query 'containerServices[0].state' --output text)" = "READY" ]; do
    sleep 15
  done
fi

echo "==> pushing images"
push() {  # label image -> the ":service.label.N" reference Lightsail assigns
  command aws --region "$REGION" lightsail push-container-image \
    --service-name "$SERVICE" --label "$1" --image "$2" 2>&1 \
    | sed -n 's/.*Refer to this image as "\(.*\)" in deployments.*/\1/p'
}
SERVICE_IMAGE=$(push service policygate-service:deploy)
DASHBOARD_IMAGE=$(push dashboard policygate-dashboard:deploy)
[ -n "$SERVICE_IMAGE" ] && [ -n "$DASHBOARD_IMAGE" ] || { echo "push failed" >&2; exit 1; }
echo "    service:   $SERVICE_IMAGE"
echo "    dashboard: $DASHBOARD_IMAGE"

echo "==> deploying"
containers=$(mktemp) && endpoint=$(mktemp)
trap 'rm -f "$containers" "$endpoint"' EXIT

# The service keeps its default SQLite audit store and in-memory checkpointer:
# a container service has no volume, so both are lost on redeploy. That is the
# right trade for a demo and the wrong one for anything else -- point
# POLICYGATE_AUDIT_URL and POLICYGATE_CHECKPOINT_URL at a managed database when
# a pending approval has to survive a restart.
SERVICE_IMAGE="$SERVICE_IMAGE" DASHBOARD_IMAGE="$DASHBOARD_IMAGE" \
CAP="$CAP" RUN_CACHE="$RUN_CACHE" \
OPENAI_API_KEY="$OPENAI_API_KEY" OPENAI_API_BASE="$OPENAI_API_BASE" \
POLICYGATE_DASHBOARD_PASSWORD="$POLICYGATE_DASHBOARD_PASSWORD" \
python3 - > "$containers" <<'PY'
import json, os
json.dump({
    "service": {
        "image": os.environ["SERVICE_IMAGE"],
        "environment": {
            "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
            "OPENAI_API_BASE": os.environ["OPENAI_API_BASE"],
            "POLICYGATE_SESSION_USD_CAP": os.environ["CAP"],
        },
        "ports": {"8000": "HTTP"},
    },
    "dashboard": {
        "image": os.environ["DASHBOARD_IMAGE"],
        "environment": {
            "POLICYGATE_DASHBOARD_PASSWORD": os.environ["POLICYGATE_DASHBOARD_PASSWORD"],
            "POLICYGATE_RUN_CACHE": os.environ["RUN_CACHE"],
            # Containers in one deployment reach each other on localhost.
            "POLICYGATE_SERVICE_URL": "http://localhost:8000",
        },
        "ports": {"8501": "HTTP"},
    },
}, os.sys.stdout)
PY

cat > "$endpoint" <<JSON
{
  "containerName": "dashboard",
  "containerPort": 8501,
  "healthCheck": {
    "path": "/_stcore/health",
    "intervalSeconds": 30,
    "timeoutSeconds": 5,
    "healthyThreshold": 2,
    "unhealthyThreshold": 4,
    "successCodes": "200-299"
  }
}
JSON

aws lightsail create-container-service-deployment \
  --service-name "$SERVICE" \
  --containers "file://$containers" \
  --public-endpoint "file://$endpoint" >/dev/null

echo "==> deploying; this takes a few minutes"
until [ "$(aws lightsail get-container-services --service-name "$SERVICE" \
            --query 'containerServices[0].currentDeployment.state' --output text)" = "ACTIVE" ]; do
  sleep 20
done

URL=$(aws lightsail get-container-services --service-name "$SERVICE" \
        --query 'containerServices[0].url' --output text)
echo
echo "live: $URL"
echo "password: the one in POLICYGATE_DASHBOARD_PASSWORD"
