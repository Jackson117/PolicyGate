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

# Lightsail runs linux/amd64. Building on an Apple Silicon Mac without this
# produces arm64 images that build and run perfectly here and are rejected at
# the push with "image does not provide linux/amd64 platform" -- after the
# build, and after the service is already created and billing.
PLATFORM=${LIGHTSAIL_PLATFORM:-linux/amd64}

echo "==> building images for $PLATFORM"
echo "    (emulated on an arm64 host: the first build is slow, then it caches)"
docker build --platform "$PLATFORM" --target service   -t policygate-service:deploy   .
docker build --platform "$PLATFORM" --target dashboard -t policygate-dashboard:deploy .

if ! aws lightsail get-container-services --service-name "$SERVICE" >/dev/null 2>&1; then
  echo "==> creating container service $SERVICE ($POWER x$SCALE in $REGION)"
  aws lightsail create-container-service \
    --service-name "$SERVICE" --power "$POWER" --scale "$SCALE" >/dev/null
fi

# Outside the branch above on purpose: a service that already exists can still
# be PENDING -- created a moment ago, or updating -- and pushing to a registry
# that is not ready fails after the upload rather than before it.
state=$(aws lightsail get-container-services --service-name "$SERVICE" \
          --query 'containerServices[0].state' --output text)
if [ "$state" != "READY" ]; then
  echo "==> service is $state; waiting for READY (a few minutes)"
  until [ "$(aws lightsail get-container-services --service-name "$SERVICE" \
              --query 'containerServices[0].state' --output text)" = "READY" ]; do
    sleep 15
  done
fi

# lightsailctl talks to Docker itself, through the Go SDK, and the SDK looks at
# DOCKER_HOST or falls back to unix:///var/run/docker.sock. It knows nothing
# about docker contexts -- so with Colima, Rancher Desktop, or Docker Desktop
# without the default-socket option, `docker build` works and the push fails
# with "Cannot connect to the Docker daemon". Hand it the endpoint the docker
# CLI is actually using.
if [ -z "${DOCKER_HOST:-}" ]; then
  endpoint=$(docker context inspect --format '{{.Endpoints.docker.Host}}' 2>/dev/null || true)
  if [ -n "$endpoint" ] && [ "$endpoint" != "unix:///var/run/docker.sock" ]; then
    export DOCKER_HOST="$endpoint"
    echo "==> DOCKER_HOST=$DOCKER_HOST (from the '$(docker context show)' context)"
  fi
fi

echo "==> pushing images"
push() {  # label image -> the ":service.label.N" reference Lightsail assigns
  # The output is captured rather than piped so that a failure reaches the log.
  # Piping it straight into sed prints only the matching line, which on failure
  # is no line at all -- the error that explains it is dropped, and all that is
  # left is a non-zero exit from a script with `set -e`.
  local out
  if ! out=$(command aws --region "$REGION" lightsail push-container-image \
               --service-name "$SERVICE" --label "$1" --image "$2" 2>&1); then
    echo "push of $2 failed:" >&2
    echo "$out" >&2
    return 1
  fi
  printf '%s\n' "$out" \
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
