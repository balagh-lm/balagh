# Balagh on AWS: continued pretraining on one H100

Everything here runs from your PC with your own AWS login. Nothing changes your account unless the command has `--yes`; without it, the command prints what it would do. The training instance runs by itself (Auto Scaling group of size 1, Spot), so your PC can be off once it is launched.

## Files

| File | What it does |
|---|---|
| `launch.py` | check, request quotas, one-time setup, budgets, launch, status, logs, stop |
| `bootstrap.sh` | runs on the instance: fail-closed guards, pinned installs, data download, smoke, pilot or main run, self-shutdown |
| `train_cpt.py` | the trainer: fp32 master + bf16, Gated DeltaNet kernels, chunked loss, checkpoints to S3, Spot-safe |
| `pilot_report.py` | compares the pilot's learning rates and prints the recommended one plus measured throughput |

## One-time steps (about 15 minutes, then waiting on AWS)

1. Log in on this PC under a profile named `ownova`, then confirm:

```bash
aws configure sso --profile ownova
```

```bash
python launch.py check --all-regions
```

`check` shows the account plan (it must be **Paid** for Activate credits), the GPU quotas, live Spot prices, the AMI, and whether the HF token is stored.

2. Store the Hugging Face write token where only the training instance can read it (you type the token; it never passes through Claude):

```bash
aws ssm put-parameter --name /balagh/hf_token --type SecureString --value YOUR_HF_TOKEN --profile ownova --region us-west-2
```

3. File the small quota requests (16 vCPU of P On-Demand, 16 of P Spot, 8 of G), us-west-2 first:

```bash
python launch.py request-quotas --yes
```

4. Create the S3 bucket, instance role and security group, then the two cost budgets:

```bash
python launch.py setup --yes
```

```bash
python launch.py budget --yes
```

## Runs (after the quota is approved and the token shards exist on the Hub)

Smoke test, about 45 minutes and $5: installs, kernels, 19 steps, a checkpoint, a resume, then stops itself.

```bash
python launch.py launch --mode smoke --run smoke-1 --yes
```

Pilot, 3 learning rates x 150M tokens, about 5-8 h:

```bash
python launch.py launch --mode pilot --run pilot-1 --yes
```

Main run (set the learning rate and token budget from the pilot report):

```bash
python launch.py launch --mode main --run balagh-nano-v0 --train-env "LR=6e-5 TOTAL_TOKENS=8e9" --cost-cap 650 --yes
```

Watch it, read its log, or stop it:

```bash
python launch.py status --run balagh-nano-v0
```

```bash
python launch.py logs --run balagh-nano-v0
```

```bash
python launch.py stop --run balagh-nano-v0 --yes
```

## Safety rails

- Quotas allow one GPU instance, nothing bigger.
- The instance counts its own hours in S3 at the On-Demand price (an upper bound on Spot) and stops training at `--cost-cap`.
- It refuses to start a 7th launch of the same run, stops after two crashes, and scales its group to 0 when done, paused or capped.
- Checkpoints go to S3 every 30 minutes and immediately on a Spot interruption notice. A replacement instance resumes from the newest one.
- Budgets email muneeb@ownova.org at 50/80/100% of $700 gross spend and of $5 of real card charges. Budget data lags 8-12 hours, so the in-instance cap is the real brake.
