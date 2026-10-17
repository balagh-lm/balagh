"""Balagh AWS launcher. Run on your own PC with your AWS login (profile "ownova" by default).

Read-only:
  python launch.py check [--all-regions]      identity, account plan, GPU quotas, Spot prices, AMI, token, default VPC
  python launch.py status --run RUN           Auto Scaling group, instance, cost hours, last metrics
  python launch.py logs --run RUN             tail of the newest instance bootstrap log
Changes your account (each needs --yes; without it the command only prints what it would do):
  python launch.py request-quotas --yes       file the small GPU quota increases (us-west-2 first)
  python launch.py setup --yes                S3 bucket, instance role/profile, security group (one-time, per region)
  python launch.py budget --yes               two cost budgets: $700 gross alert, $5 real-charge alert
  python launch.py launch --mode smoke|pilot|main --run RUN [--market spot|ondemand] --yes
  python launch.py stop --run RUN --yes       graceful: the trainer checkpoints, then the group scales to 0 (add --now to force)

The HF token is never handled here: store it yourself with
  aws ssm put-parameter --name /balagh/hf_token --type SecureString --value <token> --profile ownova --region <region>
"""
import argparse, base64, json, re, shlex, sys, time, datetime, statistics, urllib.request

import boto3
from botocore.exceptions import ClientError

REGIONS = ["us-west-2", "us-east-2", "us-east-1"]   # us-west-2 first: Spot p5.4xlarge ~68% off there vs 9-15% in us-east-1/2 (Spot Advisor, Oct 2026)
ON_DEMAND = {"p5.4xlarge": 6.88, "g6e.xlarge": 1.861, "g6e.2xlarge": 2.24208, "g7e.2xlarge": 3.36312}   # single-GPU types only
VCPUS = {"p5.4xlarge": 16, "g6e.xlarge": 4, "g6e.2xlarge": 8, "g7e.2xlarge": 8}
QUOTA_ASKS = [   # (quota name regex, desired vCPUs, why)
    (r"^Running On-Demand P instances$", 16, "one p5.4xlarge (1x H100) for continued pretraining"),
    (r"^All P5? ?Spot Instance Requests$|^All P5 Spot Instance Requests$", 16, "the same p5.4xlarge on Spot"),
    (r"^Running On-Demand G and VT instances$", 8, "one g6e.2xlarge (1x L40S) for evaluation and fallback"),
]
AMI_PARAMS = [
    "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id",
    "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id",
]
ROLE, PROFILE_NAME, SG_NAME, LT_NAME = "BalaghTrainRole", "BalaghTrainProfile", "balagh-train-sg", "balagh-train"
GH_REPO = "balagh-lm/balagh"; REPO_RAW = f"https://raw.githubusercontent.com/{GH_REPO}"
RESET_ON_LAUNCH = ["launches.txt", "failures.txt", "STOPPED_REASON.txt", "COST_CAP_HIT", "STOP_REQUEST"]   # hours.txt is kept: the cap is per run

def session(a, region=None): return boto3.Session(profile_name=a.profile, region_name=region or a.region)
def acct(s): return s.client("sts").get_caller_identity()["Account"]
def bucket_name(s, region): return f"balagh-train-{acct(s)}-{region}"
def say(*x): print(*x, flush=True)
def need_yes(a, what):
    if not a.yes: say(f"[dry run] would {what}. Re-run with --yes to do it."); return False
    return True

def ec2_quotas(region_session):
    sq = region_session.client("service-quotas")
    return [q for page in sq.get_paginator("list_service_quotas").paginate(ServiceCode="ec2") for q in page["Quotas"]]

# ------------------------------------------------------------------ check (read-only)
def cmd_check(a):
    s = session(a); sts = s.client("sts").get_caller_identity()
    say(f"identity: account {sts['Account']} as {sts['Arn']}")
    try:
        st = s.client("freetier", region_name="us-east-1").get_account_plan_state()
        say(f"account plan: {st.get('accountPlanType')} status {st.get('accountPlanStatus')} remaining credits {st.get('accountPlanRemainingCredits')}")
    except Exception as e:
        say(f"account plan: not readable ({str(e)[:90]}); check Billing > Account plan: Activate credits require the Paid plan")
    for region in REGIONS if a.all_regions else [a.region]:
        rs = session(a, region); say(f"\n== {region}")
        try:
            for q in sorted((q for q in ec2_quotas(rs) if re.search(r"(On-Demand (P|G and VT) instances|(P|P5|G and VT) ?Spot Instance Requests|P5 Capacity Blocks)", q["QuotaName"])), key=lambda q: q["QuotaName"]):
                say(f"   quota {q['QuotaName']:55s} {q['QuotaCode']:12s} = {q['Value']:.0f}")
        except Exception as e: say("   quotas: error", str(e)[:120])
        ec2 = rs.client("ec2")
        try:
            off = ec2.describe_instance_type_offerings(LocationType="availability-zone", Filters=[{"Name": "instance-type", "Values": list(ON_DEMAND)}])["InstanceTypeOfferings"]
            zones = {}
            for o in off: zones.setdefault(o["InstanceType"], []).append(o["Location"])
            say("   offered in zones:", {k: sorted(v) for k, v in zones.items()})
            hist = ec2.describe_spot_price_history(InstanceTypes=list(ON_DEMAND), ProductDescriptions=["Linux/UNIX"],
                                                   StartTime=datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1))["SpotPriceHistory"]
            by = {}
            for h in hist: by.setdefault(h["InstanceType"], []).append(float(h["SpotPrice"]))
            for k, v in sorted(by.items()):
                say(f"   spot {k:12s} min ${min(v):.2f}/h median ${statistics.median(v):.2f}/h (on-demand ${ON_DEMAND.get(k, 0):.2f})")
        except Exception as e: say("   ec2: error", str(e)[:120])
        ssm = rs.client("ssm")
        for p in AMI_PARAMS:
            try: say(f"   AMI {p.split('/')[-3]}: {ssm.get_parameter(Name=p)['Parameter']['Value']}"); break
            except ClientError: continue
        try:
            present = bool(ssm.describe_parameters(ParameterFilters=[{"Key": "Name", "Values": ["/balagh/hf_token"]}])["Parameters"])
            say(f"   SSM /balagh/hf_token: {'present' if present else 'MISSING'}")
        except Exception as e: say("   SSM:", str(e)[:100])
        try:
            v = ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])["Vpcs"]
            say(f"   default VPC: {v[0]['VpcId'] if v else 'none'}")
        except Exception as e: say("   VPC:", str(e)[:100])

# ------------------------------------------------------------------ quotas
def cmd_request_quotas(a):
    for region in REGIONS:
        rs = session(a, region); sq = rs.client("service-quotas"); quotas = ec2_quotas(rs)
        for rx, want, why in QUOTA_ASKS:
            matches = [q for q in quotas if re.search(rx, q["QuotaName"])]
            if "Spot" in rx and len(matches) > 1: matches = [q for q in matches if "P5" in q["QuotaName"]] or matches   # prefer a P5-specific quota if AWS lists one
            if not matches: say(f"{region}: no quota matches {rx}"); continue
            q = matches[0]
            if q["Value"] >= want: say(f"{region} {q['QuotaName']}: already {q['Value']:.0f}"); continue
            if need_yes(a, f"request {q['QuotaName']} ({q['QuotaCode']}) = {want} in {region} for {why}"):
                try:
                    r = sq.request_service_quota_increase(ServiceCode="ec2", QuotaCode=q["QuotaCode"], DesiredValue=float(want))
                    say(f"{region} {q['QuotaName']}: requested {want}, status {r['RequestedQuota']['Status']}, case {r['RequestedQuota'].get('CaseId', '-')}")
                except ClientError as e: say(f"{region} {q['QuotaName']}: {e.response['Error']['Message'][:150]}")

# ------------------------------------------------------------------ one-time setup
def cmd_setup(a):
    s = session(a); account = acct(s); b = bucket_name(s, a.region)
    s3, iam, ec2 = s.client("s3"), s.client("iam"), s.client("ec2")
    if need_yes(a, f"create S3 bucket {b} (private, encrypted, objects expire after 60 days)"):
        try:
            kw = {} if a.region == "us-east-1" else {"CreateBucketConfiguration": {"LocationConstraint": a.region}}
            s3.create_bucket(Bucket=b, **kw); say("bucket created", b)
        except ClientError as e:
            if e.response["Error"]["Code"] != "BucketAlreadyOwnedByYou": raise
            say("bucket exists", b)
        s3.put_public_access_block(Bucket=b, PublicAccessBlockConfiguration={k: True for k in ["BlockPublicAcls", "IgnorePublicAcls", "BlockPublicPolicy", "RestrictPublicBuckets"]})
        s3.put_bucket_lifecycle_configuration(Bucket=b, LifecycleConfiguration={"Rules": [{"ID": "expire-60d", "Status": "Enabled", "Filter": {"Prefix": ""}, "Expiration": {"Days": 60}, "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 2}}]})
    policy = {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": f"arn:aws:s3:::balagh-train-{account}-*"},
        {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:AbortMultipartUpload"], "Resource": f"arn:aws:s3:::balagh-train-{account}-*/*"},
        {"Effect": "Allow", "Action": ["ssm:GetParameter"], "Resource": f"arn:aws:ssm:*:{account}:parameter/balagh/*"},
        {"Effect": "Allow", "Action": ["kms:Decrypt"], "Resource": "*", "Condition": {"StringLike": {"kms:ViaService": "ssm.*.amazonaws.com"}}},
        {"Effect": "Allow", "Action": ["autoscaling:UpdateAutoScalingGroup", "autoscaling:SetDesiredCapacity"], "Resource": f"arn:aws:autoscaling:*:{account}:autoScalingGroup:*:autoScalingGroupName/balagh-*"},
    ]}
    if need_yes(a, f"create IAM role {ROLE} + instance profile {PROFILE_NAME} (balagh-train-* buckets, SSM /balagh/*, scale balagh-* groups, Session Manager)"):
        trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
        try: iam.create_role(RoleName=ROLE, AssumeRolePolicyDocument=json.dumps(trust), Description="Balagh training instances"); say("role created")
        except ClientError as e:
            if e.response["Error"]["Code"] != "EntityAlreadyExists": raise
        iam.put_role_policy(RoleName=ROLE, PolicyName="balagh-train", PolicyDocument=json.dumps(policy))
        iam.attach_role_policy(RoleName=ROLE, PolicyArn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore")
        try: iam.create_instance_profile(InstanceProfileName=PROFILE_NAME); say("instance profile created")
        except ClientError as e:
            if e.response["Error"]["Code"] != "EntityAlreadyExists": raise
        prof = iam.get_instance_profile(InstanceProfileName=PROFILE_NAME)["InstanceProfile"]
        if not any(r["RoleName"] == ROLE for r in prof["Roles"]): iam.add_role_to_instance_profile(InstanceProfileName=PROFILE_NAME, RoleName=ROLE)
    if need_yes(a, f"create security group {SG_NAME} in the default VPC of {a.region} (no inbound rules; access via Session Manager)"):
        vpc = ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])["Vpcs"][0]["VpcId"]
        try: say("security group", ec2.create_security_group(GroupName=SG_NAME, Description="Balagh training: outbound only", VpcId=vpc)["GroupId"])
        except ClientError as e:
            if e.response["Error"]["Code"] != "InvalidGroup.Duplicate": raise
            say("security group exists")

# ------------------------------------------------------------------ budgets
def cmd_budget(a):
    s = session(a); account = acct(s); bud = s.client("budgets", region_name="us-east-1")
    for name, limit, include_credit, note in [("balagh-gross", 700, False, "gross usage before credits"), ("balagh-card-charges", 5, True, "charges left after credits, i.e. real card charges")]:
        if not need_yes(a, f"create budget {name}: ${limit}/month on {note}, email {a.email} at 50/80/100%"): continue
        try:
            bud.create_budget(AccountId=account, Budget={"BudgetName": name, "BudgetLimit": {"Amount": str(limit), "Unit": "USD"}, "TimeUnit": "MONTHLY", "BudgetType": "COST",
                              "CostTypes": {"IncludeCredit": include_credit, "IncludeRefund": include_credit, "IncludeTax": True, "IncludeSupport": True, "UseBlended": False}},
                              NotificationsWithSubscribers=[{"Notification": {"NotificationType": "ACTUAL", "ComparisonOperator": "GREATER_THAN", "Threshold": t, "ThresholdType": "PERCENTAGE"},
                                                             "Subscribers": [{"SubscriptionType": "EMAIL", "Address": a.email}]} for t in (50, 80, 100)])
            say("budget created", name)
        except ClientError as e: say(name, e.response["Error"]["Message"][:150])

# ------------------------------------------------------------------ launch / status / stop
def resolve_sha(ref):
    """A branch or tag resolved to the full commit SHA, so every Spot replacement runs identical code."""
    if re.fullmatch(r"[0-9a-f]{40}", ref): return ref
    with urllib.request.urlopen(urllib.request.Request(f"https://api.github.com/repos/{GH_REPO}/commits/{ref}", headers={"Accept": "application/vnd.github+json", "User-Agent": "balagh-launch"}), timeout=20) as r:
        return json.load(r)["sha"]

def user_data(a, b, asg, sha):
    env = {"RUN": a.run, "MODE": a.mode, "BUCKET": b, "REGION": a.region, "ASG_NAME": asg, "PRICE_PER_HOUR": str(ON_DEMAND[a.instance]),
           "COST_CAP_USD": str(a.cost_cap), "MAX_LAUNCHES": str(a.max_launches), "MAX_HOURS": str(a.max_hours), "REPO_REF": sha,
           "TOKENS_REPO": a.tokens_repo, "HF_CKPT_REPO": a.ckpt_repo, "TRAIN_ENV": a.train_env, "PILOT_LRS": a.pilot_lrs, "PILOT_TOKENS": a.pilot_tokens}
    url = f"{REPO_RAW}/{sha}/phase0/aws/bootstrap.sh"
    lines = ["#!/bin/bash"] + [f"export {k}={shlex.quote(v)}" for k, v in env.items()] + [
        "export AWS_DEFAULT_REGION=\"$REGION\"",
        f"curl -fsSL --retry 5 --retry-all-errors --retry-delay 5 {shlex.quote(url)} -o /root/balagh-bootstrap.sh && bash /root/balagh-bootstrap.sh",
        "# bootstrap.sh powers the instance off itself; reaching this line means the fetch failed or it aborted early",
        'aws autoscaling update-auto-scaling-group --auto-scaling-group-name "$ASG_NAME" --min-size 0 --desired-capacity 0',
        "shutdown -h now"]
    return "\n".join(lines) + "\n"

def cmd_launch(a):
    if a.instance not in ON_DEMAND: sys.exit(f"--instance must be a single-GPU type: {sorted(ON_DEMAND)}")
    s = session(a); ec2, asc, ssm, s3, iam = s.client("ec2"), s.client("autoscaling"), s.client("ssm"), s.client("s3"), s.client("iam")
    b = bucket_name(s, a.region); asg = f"balagh-{a.run}"
    # ---- preflight: everything an unattended instance needs, checked before anything is billed
    try: sha = resolve_sha(a.ref)
    except Exception as e: sys.exit(f"cannot resolve --ref {a.ref} on GitHub ({e})")
    url = f"{REPO_RAW}/{sha}/phase0/aws/bootstrap.sh"
    try: urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=20)
    except Exception as e: sys.exit(f"bootstrap not reachable at {url} ({e}); push phase0/aws to GitHub first")
    try: s3.head_bucket(Bucket=b)
    except ClientError: sys.exit(f"bucket {b} missing: run `setup --yes --region {a.region}`")
    try: iam.get_instance_profile(InstanceProfileName=PROFILE_NAME)
    except ClientError: sys.exit("instance profile missing: run `setup --yes`")
    try: ssm.get_parameter(Name="/balagh/hf_token")
    except ClientError: sys.exit(f"/balagh/hf_token missing in SSM {a.region}: store it with `aws ssm put-parameter ...` (see README)")
    try:
        s3.head_object(Bucket=b, Key=f"runs/{a.run}/state/DONE"); sys.exit(f"run {a.run} is already DONE: choose a new --run")
    except ClientError: pass
    need = VCPUS[a.instance]; fam = "P" if a.instance.startswith("p") else "G and VT"
    qname = (rf"^All {fam}5? ?Spot Instance Requests$|^All P5 Spot Instance Requests$" if a.market == "spot" else rf"^Running On-Demand {fam} instances$")
    qs = [q for q in ec2_quotas(s) if re.search(qname, q["QuotaName"])]
    if qs and max(q["Value"] for q in qs) < need: sys.exit(f"quota too small for {a.instance} ({a.market}): {[(q['QuotaName'], q['Value']) for q in qs]}, need {need} vCPU")
    ami = None
    for p in AMI_PARAMS:
        try: ami = ssm.get_parameter(Name=p)["Parameter"]["Value"]; break
        except ClientError: pass
    if not ami: sys.exit("no AWS Deep Learning base GPU AMI found in this region")
    sg = ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": [SG_NAME]}])["SecurityGroups"]
    if not sg: sys.exit("security group missing: run `setup --yes`")
    subnets = [x["SubnetId"] for x in ec2.describe_subnets(Filters=[{"Name": "default-for-az", "Values": ["true"]}])["Subnets"]]
    est = ON_DEMAND[a.instance]
    say(f"run {a.run}: {a.mode} on {a.instance} ({a.market}) in {a.region}; code {sha[:12]}; AMI {ami}; cost cap ${a.cost_cap} "
        f"(= {a.cost_cap / est:.0f} h at on-demand ${est}/h, an upper bound on Spot); max {a.max_launches} launches; train env: {a.train_env or '(defaults)'}")
    if not need_yes(a, f"create launch template {LT_NAME} and Auto Scaling group {asg} with desired capacity 1 (this starts a billed GPU instance)"): return
    for k in RESET_ON_LAUNCH:
        try: s3.delete_object(Bucket=b, Key=f"runs/{a.run}/state/{k}")
        except ClientError: pass
    data = {"ImageId": ami, "InstanceType": a.instance, "IamInstanceProfile": {"Name": PROFILE_NAME}, "SecurityGroupIds": [sg[0]["GroupId"]],
            "UserData": base64.b64encode(user_data(a, b, asg, sha).encode()).decode(), "InstanceInitiatedShutdownBehavior": "terminate",
            "MetadataOptions": {"HttpTokens": "required", "HttpPutResponseHopLimit": 2, "HttpEndpoint": "enabled"},
            "BlockDeviceMappings": [{"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": 200, "VolumeType": "gp3", "DeleteOnTermination": True}}],
            "TagSpecifications": [{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": asg}, {"Key": "project", "Value": "balagh"}]}]}
    try:
        ver = str(ec2.create_launch_template(LaunchTemplateName=LT_NAME, LaunchTemplateData=data)["LaunchTemplate"]["LatestVersionNumber"])
    except ClientError as e:
        if "AlreadyExists" not in e.response["Error"]["Code"]: raise
        ver = str(ec2.create_launch_template_version(LaunchTemplateName=LT_NAME, LaunchTemplateData=data)["LaunchTemplateVersion"]["VersionNumber"])
    mip = {"LaunchTemplate": {"LaunchTemplateSpecification": {"LaunchTemplateName": LT_NAME, "Version": ver}, "Overrides": [{"InstanceType": a.instance}]},
           "InstancesDistribution": {"OnDemandBaseCapacity": 0, "OnDemandPercentageAboveBaseCapacity": 0 if a.market == "spot" else 100, "SpotAllocationStrategy": "price-capacity-optimized"}}
    existing = asc.describe_auto_scaling_groups(AutoScalingGroupNames=[asg])["AutoScalingGroups"]
    if existing:
        asc.update_auto_scaling_group(AutoScalingGroupName=asg, MixedInstancesPolicy=mip, MinSize=0, MaxSize=1, DesiredCapacity=1)
    else:
        asc.create_auto_scaling_group(AutoScalingGroupName=asg, MixedInstancesPolicy=mip, MinSize=0, MaxSize=1, DesiredCapacity=1, VPCZoneIdentifier=",".join(subnets),
                                      Tags=[{"Key": "project", "Value": "balagh", "PropagateAtLaunch": True}])
    say(f"group {asg} desired 1 (launch template {LT_NAME} v{ver}). Follow with: python launch.py status --run {a.run} --region {a.region}")

def s3_text(s, b, key):
    try: return s.client("s3").get_object(Bucket=b, Key=key)["Body"].read().decode()
    except ClientError: return None

def cmd_status(a):
    s = session(a); b = bucket_name(s, a.region); asg = f"balagh-{a.run}"
    g = s.client("autoscaling").describe_auto_scaling_groups(AutoScalingGroupNames=[asg])["AutoScalingGroups"]
    if g:
        g = g[0]; say(f"group {asg}: desired {g['DesiredCapacity']}, instances {[(i['InstanceId'], i['LifecycleState'], i.get('InstanceType')) for i in g['Instances']]}")
        for x in s.client("autoscaling").describe_scaling_activities(AutoScalingGroupName=asg, MaxRecords=3)["Activities"]:
            say(f"   {x['StartTime']:%m-%d %H:%M} {x['StatusCode']}: {x['Description'][:90]} {x.get('StatusMessage', '')[:120]}")
    else: say(f"group {asg}: none")
    for k in ["launches.txt", "hours.txt", "failures.txt", "DONE", "COST_CAP_HIT", "STOP_REQUEST", "STOPPED_REASON.txt"]:
        v = s3_text(s, b, f"runs/{a.run}/state/{k}")
        if v: say(f"   {k}: {v.strip()}")
    h = s3_text(s, b, f"runs/{a.run}/state/hours.txt")
    if h: say(f"   cost so far (at on-demand ${ON_DEMAND.get(a.instance, 0)}/h, an upper bound on Spot): ${float(h) * ON_DEMAND.get(a.instance, 0):.0f}")
    m = s3_text(s, b, f"runs/{a.run}/logs/{a.run}-metrics.jsonl")
    if m:
        lines = [json.loads(x) for x in m.strip().splitlines()[-300:] if x.strip()]
        for x in [y for y in lines if "loss" in y][-3:] + [y for y in lines if "val" in y][-1:]: say("   ", json.dumps(x)[:300])
    rep = s3_text(s, b, f"runs/{a.run}/pilot_report.md")
    if rep: say("\n" + rep)

def cmd_logs(a):
    s = session(a); b = bucket_name(s, a.region)
    objs = s.client("s3").list_objects_v2(Bucket=b, Prefix=f"runs/{a.run}/logs/bootstrap-").get("Contents", [])
    if not objs: say("no bootstrap logs yet"); return
    last = max(objs, key=lambda o: o["LastModified"])
    say(f"== {last['Key']} ({last['LastModified']:%m-%d %H:%M})"); say("\n".join(s3_text(s, b, last["Key"]).splitlines()[-a.lines:]))

def cmd_stop(a):
    s = session(a); asg = f"balagh-{a.run}"; b = bucket_name(s, a.region)
    if a.now:
        if need_yes(a, f"scale {asg} to 0 immediately (terminates the instance; work since the last 30-min checkpoint is lost)"):
            s.client("autoscaling").update_auto_scaling_group(AutoScalingGroupName=asg, MinSize=0, DesiredCapacity=0); say("scaled to 0")
    elif need_yes(a, f"ask run {a.run} to stop gracefully (within ~6 min it checkpoints to S3, then scales {asg} to 0)"):
        s.client("s3").put_object(Bucket=b, Key=f"runs/{a.run}/state/STOP_REQUEST", Body=time.strftime("%Y-%m-%dT%H:%M:%S").encode()); say("stop requested")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "request-quotas", "setup", "budget", "launch", "status", "logs", "stop"])
    ap.add_argument("--profile", default="ownova"); ap.add_argument("--region", default="us-west-2"); ap.add_argument("--all-regions", action="store_true")
    ap.add_argument("--yes", action="store_true", help="actually make the change"); ap.add_argument("--now", action="store_true", help="stop: force, no checkpoint")
    ap.add_argument("--run"); ap.add_argument("--mode", default="smoke", choices=["smoke", "pilot", "main"])
    ap.add_argument("--instance", default="p5.4xlarge"); ap.add_argument("--market", default="spot", choices=["spot", "ondemand"])
    ap.add_argument("--cost-cap", type=float, default=750); ap.add_argument("--max-launches", type=int, default=6); ap.add_argument("--max-hours", type=float, default=200)
    ap.add_argument("--ref", default="main"); ap.add_argument("--tokens-repo", default="balagh-lm/tokens-qwen35-v1"); ap.add_argument("--ckpt-repo", default="balagh-lm/balagh-nano-ckpt")
    ap.add_argument("--train-env", default="", help='e.g. "LR=6e-5 TOTAL_TOKENS=8e9"')
    ap.add_argument("--pilot-lrs", default="3e-5 6e-5 1e-4"); ap.add_argument("--pilot-tokens", default="150e6")
    ap.add_argument("--email", default="muneeb@ownova.org"); ap.add_argument("--lines", type=int, default=60)
    a = ap.parse_args()
    if a.cmd in ("launch", "status", "logs", "stop") and not a.run: ap.error("--run is required for " + a.cmd)
    {"check": cmd_check, "request-quotas": cmd_request_quotas, "setup": cmd_setup, "budget": cmd_budget, "launch": cmd_launch,
     "status": cmd_status, "logs": cmd_logs, "stop": cmd_stop}[a.cmd](a)
