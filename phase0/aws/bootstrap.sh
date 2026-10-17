#!/bin/bash
# Balagh training-instance bootstrap. Runs as root from EC2 user-data on an AWS Deep Learning Base GPU AMI (Ubuntu).
# The launcher prepends an env block (RUN, MODE, BUCKET, REGION, ASG_NAME, PRICE_PER_HOUR, COST_CAP_USD, MAX_LAUNCHES,
# MAX_HOURS, REPO_URL, REPO_REF (a commit SHA), TOKENS_REPO, HF_CKPT_REPO, TRAIN_ENV, PILOT_LRS, PILOT_TOKENS).
#
# Fail-closed lifecycle: refuse to start if the run is DONE, past MAX_LAUNCHES, past its cost cap, or its S3 state is
# unreadable; read the HF token from SSM (never printed); fetch the pinned code; install pinned torch / transformers /
# flash-linear-attention / causal-conv1d with retries; check the HF token can write; download the token shards to
# NVMe; restore metrics history; run MODE (smoke | pilot | main); keep logs, metrics and cost hours in S3. A heartbeat
# counts hours, enforces the cost cap (graceful stop, then forced shutdown after 30 min) and honours an operator stop
# flag. At the end the group is scaled to 0 (done, paused, capped, stopped, repeated crash) or the instance just powers
# off so the Auto Scaling group replaces a reclaimed Spot instance.
set -uo pipefail
: "${RUN:?}" "${MODE:?}" "${BUCKET:?}" "${REGION:?}" "${ASG_NAME:?}"
PRICE_PER_HOUR="${PRICE_PER_HOUR:-6.88}"; COST_CAP_USD="${COST_CAP_USD:-750}"; MAX_LAUNCHES="${MAX_LAUNCHES:-6}"
MAX_HOURS="${MAX_HOURS:-200}"; REPO_URL="${REPO_URL:-https://github.com/balagh-lm/balagh.git}"; REPO_REF="${REPO_REF:-main}"
TOKENS_REPO="${TOKENS_REPO:-balagh-lm/tokens-qwen35-v1}"; HF_CKPT_REPO="${HF_CKPT_REPO:-balagh-lm/balagh-nano-ckpt}"; TRAIN_ENV="${TRAIN_ENV:-}"
PILOT_LRS="${PILOT_LRS:-3e-5 6e-5 1e-4}"; PILOT_TOKENS="${PILOT_TOKENS:-150e6}"
export AWS_DEFAULT_REGION="$REGION" AWS_RETRY_MODE=standard AWS_MAX_ATTEMPTS=10 DEBIAN_FRONTEND=noninteractive
S3="s3://$BUCKET/runs/$RUN"
LOG=/var/log/balagh-bootstrap.log
LAUNCH_ID="$(date -u +%Y%m%dT%H%M%S)"; WORK=/home/ubuntu/balagh
exec > >(tee -a "$LOG") 2>&1
echo "== balagh bootstrap $(date -u +%FT%TZ) run=$RUN mode=$MODE ref=$REPO_REF"
aws configure set default.s3.max_concurrent_requests 64; aws configure set default.s3.multipart_chunksize 64MB; aws configure set default.s3.multipart_threshold 64MB

sync_logs() {
  aws s3 cp --only-show-errors "$LOG" "$S3/logs/bootstrap-$LAUNCH_ID.log" || true
  for f in "$WORK"/runs/*/metrics.jsonl; do
    [ -f "$f" ] && aws s3 cp --only-show-errors "$f" "$S3/logs/$(basename "$(dirname "$f")")-metrics.jsonl" || true
  done
}
stop_group() {   # scale to 0 so the group does not replace this instance, then power off (terminate)
  echo "scaling $ASG_NAME to 0: $1"
  for i in 1 2 3; do aws autoscaling update-auto-scaling-group --auto-scaling-group-name "$ASG_NAME" --min-size 0 --desired-capacity 0 && break; sleep 10; done
  echo "$1" | aws s3 cp --only-show-errors - "$S3/state/STOPPED_REASON.txt" || true
  sync_logs; sleep 5; shutdown -h now; exit 0
}
s3_get() {   # prints the object; returns 0 found, 44 missing, 2 unreadable after retries
  local out i
  for i in 1 2 3 4 5; do
    if out=$(aws s3 cp --only-show-errors "$S3/state/$1" - 2>/tmp/s3err); then printf '%s' "$out"; return 0; fi
    grep -qE "\(404\)|NoSuchKey|does not exist|Not Found" /tmp/s3err && return 44
    sleep $((i * 4))
  done
  cat /tmp/s3err; return 2
}
s3_num() {   # number at key, 0 if missing, rc 2 if unreadable or not a number
  local v rc; v=$(s3_get "$1"); rc=$?
  [ $rc -eq 44 ] && { echo 0; return 0; }
  [ $rc -ne 0 ] && return 2
  [[ "$v" =~ ^[0-9]+(\.[0-9]+)?$ ]] && { echo "$v"; return 0; }
  return 2
}
s3_put() { local i; for i in 1 2 3; do echo "$2" | aws s3 cp --only-show-errors - "$S3/state/$1" && return 0; sleep 5; done; return 1; }
retry() { local n; for n in 1 2 3 4; do "$@" && return 0; echo "retry $n/4 failed: $*"; sleep $((n * 20)); done; return 1; }
over_cap() { python3 -c "import sys; sys.exit(0 if float('$1') * float('$PRICE_PER_HOUR') >= float('$COST_CAP_USD') else 1)"; }

# ---- scratch disk: the AMI mounts instance-store NVMe at /opt/dlami/nvme; otherwise format an empty NVMe disk
for i in $(seq 1 12); do [ -d /opt/dlami/nvme ] && break; sleep 5; done
if [ -d /opt/dlami/nvme ]; then WORK=/opt/dlami/nvme/balagh
else
  DEV=$(lsblk -dpnr -o NAME,TYPE,MOUNTPOINT | awk '$2=="disk" && $3=="" && $1 ~ /nvme/ {print $1}' | while read -r d; do
        [ -z "$(lsblk -nr -o MOUNTPOINT "$d" | tr -d ' \n')" ] && ! blkid "$d" >/dev/null 2>&1 && echo "$d"; done | head -1)
  if [ -n "$DEV" ] && mkfs.ext4 -q -F "$DEV" && mkdir -p /mnt/nvme && mount "$DEV" /mnt/nvme; then WORK=/mnt/nvme/balagh; fi
fi
mkdir -p "$WORK"; cd "$WORK"; echo "work dir: $WORK ($(df -h "$WORK" | tail -1 | awk '{print $2}'))"

# ---- fail-closed guards
s3_get DONE >/dev/null; rc=$?; [ $rc -eq 0 ] && stop_group "run already DONE"; [ $rc -eq 2 ] && stop_group "S3 state unreadable (bucket/role?)"
LAUNCHES=$(s3_num launches.txt) || stop_group "launches.txt unreadable"; LAUNCHES=$((LAUNCHES + 1)); s3_put launches.txt "$LAUNCHES" || stop_group "cannot write S3 state"
[ "$LAUNCHES" -gt "$MAX_LAUNCHES" ] && stop_group "launch $LAUNCHES exceeds MAX_LAUNCHES=$MAX_LAUNCHES"
HOURS=$(s3_num hours.txt) || stop_group "hours.txt unreadable"
over_cap "$HOURS" && stop_group "cost cap reached before start ($HOURS h x \$$PRICE_PER_HOUR >= \$$COST_CAP_USD)"
echo "launch $LAUNCHES/$MAX_LAUNCHES; cumulative $HOURS h of cap \$$COST_CAP_USD at \$$PRICE_PER_HOUR/h"

# ---- heartbeat: hours (kept locally, never re-read, so an S3 blip cannot reset it), cap with grace, operator stop
(
  H="$HOURS"; CAPN=0; START=$(date +%s)
  while true; do
    sleep 360
    H=$(python3 -c "print(round(float('$H') + 0.1, 3))"); s3_put hours.txt "$H" || echo "heartbeat: could not write hours"
    STOPNOW=""
    over_cap "$H" && { STOPNOW="cost cap"; s3_put COST_CAP_HIT "$H"; }
    s3_get STOP_REQUEST >/dev/null 2>&1 && STOPNOW="operator stop"
    python3 -c "import sys; sys.exit(0 if ($(date +%s) - $START) / 3600 > float('$MAX_HOURS') + 0.5 else 1)" && STOPNOW="MAX_HOURS backstop"
    if [ -n "$STOPNOW" ]; then
      echo "heartbeat: $STOPNOW -> asking the trainer to checkpoint and stop"
      for d in "$WORK"/runs/*/; do [ -d "$d" ] && touch "$d/STOP_REQUESTED"; done
      CAPN=$((CAPN + 1))
      if [ "$CAPN" -ge 5 ]; then stop_group "$STOPNOW: grace expired, forced shutdown"; fi
    fi
    sync_logs
  done
) &
HEARTBEAT=$!

# ---- secrets (never echoed)
set +x
HF_TOKEN="$(aws ssm get-parameter --name /balagh/hf_token --with-decryption --query Parameter.Value --output text 2>/dev/null)"
[ -z "$HF_TOKEN" ] && stop_group "no /balagh/hf_token in SSM Parameter Store ($REGION)"
export HF_TOKEN HF_XET_HIGH_PERFORMANCE=1

# ---- pinned code first (cheap to fail), then software
timeout 60 nvidia-smi || stop_group "no GPU visible"
retry timeout 900 apt-get -o DPkg::Lock::Timeout=300 -qq update || echo "apt update failed (continuing)"
retry timeout 900 apt-get -o DPkg::Lock::Timeout=300 -qq install -y git python3-pip || stop_group "apt install git/pip failed"
fetch_repo() {   # works for a commit SHA, a tag or a branch
  rm -rf "$WORK/repo"; git init -q "$WORK/repo" && timeout 600 git -C "$WORK/repo" fetch -q --depth 1 "$REPO_URL" "$REPO_REF" && git -C "$WORK/repo" checkout -q FETCH_HEAD
}
[ -d "$WORK/repo/.git" ] || retry fetch_repo || stop_group "git fetch of $REPO_REF failed"
echo "repo commit: $(git -C "$WORK/repo" rev-parse HEAD)"
if [ ! -x "$WORK/venv/bin/python" ]; then
  retry timeout 600 python3 -m pip install -q --break-system-packages uv || retry timeout 600 python3 -m pip install -q uv || stop_group "uv install failed"
  retry timeout 900 uv venv -q --seed -p 3.12 "$WORK/venv" || stop_group "venv create failed"
  # torch 2.10: causal-conv1d 1.7.0 publishes prebuilt wheels up to torch 2.10 (newer torch needs an nvcc build)
  retry timeout 1800 uv pip install -q -p "$WORK/venv/bin/python" "torch==2.10.0" --index-url https://download.pytorch.org/whl/cu128 || stop_group "torch install failed"
  retry timeout 1800 uv pip install -q -p "$WORK/venv/bin/python" "transformers==5.20.1" "flash-linear-attention==0.5.2" "huggingface_hub[hf_xet]" safetensors numpy packaging ninja wheel setuptools || stop_group "pip install failed"
  timeout 2400 uv pip install -q -p "$WORK/venv/bin/python" --no-build-isolation "causal-conv1d==1.7.0" || echo "causal-conv1d install failed"
fi
PY="$WORK/venv/bin/python"
"$PY" -c "import torch, transformers, fla; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__, 'fla', fla.__version__)" || stop_group "import check failed"
if "$PY" -c "import causal_conv1d; print('causal_conv1d', causal_conv1d.__version__)"; then rm -f "$WORK/kernels.env"
else echo "WARNING: causal-conv1d unusable: using Hugging Face Hub kernels"; retry uv pip install -q -p "$PY" kernels || stop_group "no causal-conv1d and no hub kernels"; echo "USE_HUB_KERNELS=1" > "$WORK/kernels.env"; fi

if [ "$MODE" = "main" ]; then   # a bad token should fail in seconds, not at the first milestone hours later
  "$PY" -c "import os; from huggingface_hub import HfApi; HfApi(token=os.environ['HF_TOKEN']).create_repo('$HF_CKPT_REPO', private=True, exist_ok=True); print('HF write ok')" || stop_group "HF token cannot write to $HF_CKPT_REPO"
fi
retry timeout 3600 "$WORK/venv/bin/hf" download "$TOKENS_REPO" --repo-type dataset --local-dir "$WORK/tokens" --quiet || stop_group "token shard download failed"
echo "tokens on disk: $(du -sh "$WORK/tokens" | cut -f1)"

restore_metrics() {   # $1 = run name: continue the S3 history instead of overwriting it with a fragment
  mkdir -p "$WORK/runs/$1"
  [ -s "$WORK/runs/$1/metrics.jsonl" ] || aws s3 cp --only-show-errors "$S3/logs/$1-metrics.jsonl" "$WORK/runs/$1/metrics.jsonl" 2>/dev/null || rm -f "$WORK/runs/$1/metrics.jsonl"
}
run_train() {   # $1 = run name, rest = KEY=VAL pairs
  local name="$1"; shift
  restore_metrics "$name"
  local kenv=""; [ -f "$WORK/kernels.env" ] && kenv="$(cat "$WORK/kernels.env")"
  local expect="EXPECT_RESUME=0"; [ "$MODE" = "main" ] && [ "$LAUNCHES" -gt 1 ] && expect="EXPECT_RESUME=1"
  env $kenv $expect $TRAIN_ENV "$@" RUN="$name" DATA_DIR="$WORK/tokens" OUT_DIR="$WORK/runs" S3_URI="s3://$BUCKET/runs" MAX_HOURS="$MAX_HOURS" HF_CKPT_REPO="$HF_CKPT_REPO" \
      "$PY" -u "$WORK/repo/phase0/aws/train_cpt.py" 2>&1 | tee -a "/var/log/balagh-train-$name.log"
  return "${PIPESTATUS[0]}"
}

RC=0
case "$MODE" in
  smoke)
    run_train "$RUN" SMOKE=1 HF_PUSH=0 TOTAL_TOKENS=21e6; RC=$?
    if [ $RC -eq 0 ]; then
      echo "== smoke resume test: wipe the local checkpoint, resume from S3"
      aws s3 ls "$S3/ckpt/latest.json" >/dev/null || { echo "smoke: no checkpoint in S3"; RC=1; }
    fi
    if [ $RC -eq 0 ]; then
      rm -rf "$WORK/runs/$RUN/ckpt" "$WORK/runs/$RUN/DONE"
      BEFORE=$(grep -c "resumed from" "/var/log/balagh-train-$RUN.log" || true)
      run_train "$RUN" SMOKE=1 HF_PUSH=0 TOTAL_TOKENS=42e6 ALLOW_CONFIG_CHANGE=1; RC=$?
      AFTER=$(grep -c "resumed from" "/var/log/balagh-train-$RUN.log" || true)
      [ $RC -eq 0 ] && [ "$AFTER" -le "$BEFORE" ] && { echo "smoke FAILED: the second run did not resume from S3"; RC=1; }
      [ $RC -eq 0 ] && echo "smoke PASSED: kernels, training, checkpoint to S3 and resume from S3 all work"
    fi
    ;;
  pilot)
    for lr in $PILOT_LRS; do
      arm="$RUN-lr$lr"
      if aws s3 ls "s3://$BUCKET/runs/$arm/DONE" >/dev/null 2>&1; then echo "arm $arm done"; restore_metrics "$arm"; continue; fi
      run_train "$arm" LR="$lr" TOTAL_TOKENS="$PILOT_TOKENS" SCHEDULE=constant WARMUP_FRAC=0.05 EVAL_EVERY=40 HF_PUSH=0 MILESTONE_TOKENS=1e18; RC=$?
      [ $RC -eq 0 ] && aws s3 cp --only-show-errors "$WORK/runs/$arm/DONE" "s3://$BUCKET/runs/$arm/DONE"
      [ $RC -ne 0 ] && break
    done
    if [ $RC -eq 0 ]; then
      sync_logs
      "$PY" "$WORK/repo/phase0/aws/pilot_report.py" "$WORK/runs" $PILOT_LRS > /tmp/pilot_report.md; PR=$?
      cat /tmp/pilot_report.md; aws s3 cp --only-show-errors /tmp/pilot_report.md "$S3/pilot_report.md"
      [ $PR -ne 0 ] && echo "pilot report: no recommendation (see report)"
    fi
    ;;
  main)
    run_train "$RUN"; RC=$?
    ;;
  *) stop_group "unknown MODE $MODE" ;;
esac
kill $HEARTBEAT 2>/dev/null; sync_logs
echo "== trainer exit code $RC"
case $RC in
  0) s3_put DONE "$(date -u +%FT%TZ)"; stop_group "finished ($MODE)";;
  3) H=$(s3_num hours.txt || echo 0)
     over_cap "$H" && stop_group "cost cap reached"
     s3_get STOP_REQUEST >/dev/null 2>&1 && stop_group "operator stop"
     echo "interrupted (Spot or signal): powering off so the group can replace the instance"; sleep 5; shutdown -h now;;
  4) stop_group "MAX_HOURS reached: paused for review";;
  *) FAILS=$(s3_num failures.txt) || FAILS=1; FAILS=$((FAILS + 1)); s3_put failures.txt "$FAILS"
     [ "$FAILS" -ge 2 ] && stop_group "trainer failed $FAILS times (last exit $RC)"
     echo "trainer failed (exit $RC); one retry allowed"; sleep 5; shutdown -h now;;
esac
