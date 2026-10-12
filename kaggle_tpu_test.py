# Balagh Phase 0: Kaggle TPU v5e-8 feasibility test
#
# Purpose: in one Kaggle session (~20 min), answer four questions with numbers:
#   1. Which TPU do we actually get, how many chips, how much HBM per chip?
#   2. What matmul throughput (TFLOP/s per chip) can JAX reach here? (sets the tokens/week estimate)
#   3. Does a sharded training step of a small transformer run, and at what tokens/s?
#   4. Does checkpoint save + restore work, so a 9-hour session cap is survivable?
#
# How to run on Kaggle:
#   New notebook -> Settings -> Accelerator: "TPU VM v5e-8" (or whatever TPU option is listed)
#   Add one code cell:   !pip -q install -U "jax[tpu]" flax optax orbax-checkpoint -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
#   Then paste this whole file into a cell and run. Copy the printed REPORT block into the Phase 0 log.
#
# Nothing here downloads a model; it is a pure feasibility probe.

import os, time, json, math, sys
import numpy as np

REPORT = {}

# ---------------------------------------------------------------- 1. devices
import jax, jax.numpy as jnp
devs = jax.devices()
REPORT["jax_version"] = jax.__version__
REPORT["device_count"] = len(devs)
REPORT["device_kind"] = devs[0].device_kind if devs else "none"
REPORT["platform"] = devs[0].platform if devs else "none"
try:
    ms = devs[0].memory_stats() or {}
    REPORT["hbm_per_chip_GB"] = round(ms.get("bytes_limit", 0) / 1e9, 1)
except Exception as e:  # memory_stats is not available on every backend
    REPORT["hbm_per_chip_GB"] = f"unavailable ({type(e).__name__})"
print("devices:", REPORT["device_count"], REPORT["device_kind"], "HBM/chip:", REPORT["hbm_per_chip_GB"])

# ---------------------------------------------------------------- 2. matmul throughput
# Peak bf16 for a v5e chip is ~197 TFLOP/s; v6e ~918; v4 ~275. We measure what we get.
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
mesh = Mesh(np.array(devs), axis_names=("d",))
N = 8192
a = jax.device_put(jnp.ones((len(devs) * N, N), jnp.bfloat16), NamedSharding(mesh, P("d", None)))
b = jax.device_put(jnp.ones((N, N), jnp.bfloat16), NamedSharding(mesh, P(None, None)))
mm = jax.jit(lambda x, y: x @ y)
mm(a, b).block_until_ready()
t0 = time.perf_counter(); iters = 10
for _ in range(iters):
    out = mm(a, b)
out.block_until_ready()
dt = (time.perf_counter() - t0) / iters
flops = 2.0 * (len(devs) * N) * N * N
REPORT["matmul_TFLOPs_per_chip"] = round(flops / dt / len(devs) / 1e12, 1)
print("matmul TFLOP/s per chip:", REPORT["matmul_TFLOPs_per_chip"])

# ---------------------------------------------------------------- 3. sharded training step of a small transformer
import flax.linen as nn, optax

class Block(nn.Module):
    d: int; heads: int
    @nn.compact
    def __call__(self, x):
        h = nn.LayerNorm()(x)
        h = nn.MultiHeadDotProductAttention(num_heads=self.heads, qkv_features=self.d, dtype=jnp.bfloat16)(h, h)
        x = x + h
        h = nn.LayerNorm()(x)
        h = nn.Dense(4 * self.d, dtype=jnp.bfloat16)(h); h = nn.gelu(h); h = nn.Dense(self.d, dtype=jnp.bfloat16)(h)
        return x + h

class LM(nn.Module):
    vocab: int = 32768; d: int = 1024; layers: int = 12; heads: int = 16; seq: int = 1024
    @nn.compact
    def __call__(self, tokens):
        x = nn.Embed(self.vocab, self.d, dtype=jnp.bfloat16)(tokens)
        x = x + self.param("pos", nn.initializers.normal(0.02), (self.seq, self.d)).astype(jnp.bfloat16)
        for _ in range(self.layers):
            x = Block(self.d, self.heads)(x)
        x = nn.LayerNorm()(x)
        return nn.Dense(self.vocab, dtype=jnp.bfloat16)(x)

# per-chip batch sized from measured HBM: logits are BATCH*SEQ*VOCAB*4 bytes in fp32 plus grads
hbm = REPORT["hbm_per_chip_GB"] if isinstance(REPORT["hbm_per_chip_GB"], float) else 16.0
PER_CHIP = 8 if hbm >= 30 else (4 if hbm >= 20 else (2 if hbm >= 12 else 1))
SEQ, BATCH = 1024, PER_CHIP * len(devs)
REPORT["per_chip_batch"] = PER_CHIP
model = LM(seq=SEQ)
key = jax.random.PRNGKey(0)
params = model.init(key, jnp.zeros((1, SEQ), jnp.int32))
n_params = sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(params))
REPORT["test_model_params_M"] = round(n_params / 1e6, 1)
tx = optax.adamw(2e-4, weight_decay=0.1)
opt_state = tx.init(params)
data_shard = NamedSharding(mesh, P("d", None))
rep = NamedSharding(mesh, P())
params = jax.device_put(params, rep); opt_state = jax.device_put(opt_state, rep)   # replicated params (small model); FSDP comes later

def loss_fn(p, toks):
    logits = model.apply(p, toks[:, :-1]).astype(jnp.float32)
    return optax.softmax_cross_entropy_with_integer_labels(logits, toks[:, 1:]).mean()

@jax.jit
def step(p, s, toks):
    loss, g = jax.value_and_grad(loss_fn)(p, toks)
    upd, s = tx.update(g, s, p)
    return optax.apply_updates(p, upd), s, loss

def run_steps(batch, n=20):
    global params, opt_state
    toks = jax.device_put(jax.random.randint(key, (batch, SEQ + 1), 0, 32768), data_shard)
    params, opt_state, loss = step(params, opt_state, toks); loss.block_until_ready()   # compile
    t0 = time.perf_counter()
    for i in range(n):
        params, opt_state, loss = step(params, opt_state, toks)
    loss.block_until_ready()
    return (time.perf_counter() - t0) / n, loss
while True:
    try:
        dt, loss = run_steps(BATCH); break
    except Exception as e:  # XlaRuntimeError RESOURCE_EXHAUSTED on small-HBM chips
        if 'RESOURCE_EXHAUSTED' in str(e) or 'Out of memory' in str(e) or 'OOM' in str(e):
            if BATCH // len(devs) <= 1: raise
            BATCH = (BATCH // len(devs) // 2) * len(devs); REPORT["per_chip_batch"] = BATCH // len(devs)
            print(f"OOM, retrying with per-chip batch {BATCH // len(devs)}")
        else: raise
tok_per_s = BATCH * SEQ / dt
REPORT["train_tokens_per_s"] = int(tok_per_s)
REPORT["train_model_flops_util"] = round(6 * n_params * tok_per_s / (len(devs) * REPORT["matmul_TFLOPs_per_chip"] * 1e12), 3)
print(f"train step: {dt*1000:.0f} ms, {tok_per_s:,.0f} tokens/s, loss {float(loss):.3f}")

# ---------------------------------------------------------------- 4. checkpoint save / restore (session-cap survival)
import orbax.checkpoint as ocp
ckdir = os.path.abspath("/kaggle/working/ckpt_test") if os.path.isdir("/kaggle/working") else os.path.abspath("ckpt_test")
ckpt = ocp.StandardCheckpointer()
state = {"params": params, "opt_state": opt_state, "step": jnp.array(21)}
t0 = time.perf_counter(); ckpt.save(os.path.join(ckdir, "step_21"), state, force=True); ckpt.wait_until_finished()
REPORT["ckpt_save_s"] = round(time.perf_counter() - t0, 1)
t0 = time.perf_counter(); restored = ckpt.restore(os.path.join(ckdir, "step_21"), state)
REPORT["ckpt_restore_s"] = round(time.perf_counter() - t0, 1)
REPORT["ckpt_roundtrip_ok"] = bool(int(restored["step"]) == 21)
sz = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(ckdir) for f in fs)
REPORT["ckpt_size_GB"] = round(sz / 1e9, 2)
print("checkpoint:", REPORT["ckpt_save_s"], "s save,", REPORT["ckpt_restore_s"], "s restore, ok =", REPORT["ckpt_roundtrip_ok"])

# ---------------------------------------------------------------- 5. what it means for Balagh-nano
peak = {"TPU v5e": 197, "TPU v6e": 918, "TPU v4": 275, "TPU v5p": 459}
kind = REPORT["device_kind"]
peak_chip = next((v for k, v in peak.items() if k.split()[-1] in kind), None)
mfu = (REPORT["matmul_TFLOPs_per_chip"] / peak_chip) if peak_chip else None
weekly_flops = len(devs) * REPORT["matmul_TFLOPs_per_chip"] * 1e12 * 0.6 * 3600 * 20   # assume training reaches 60% of matmul throughput, 20 h/week
for name, nparams in [("2B-class (Gemma 4 E2B / Qwen3.5-2B)", 2.3e9), ("4B-class (Gemma 4 E4B / Qwen3.5-4B)", 4.5e9)]:
    REPORT[f"est_tokens_per_week_{name.split()[0]}"] = f"{weekly_flops / (6 * nparams) / 1e9:.1f}B"
REPORT["assumed_matmul_mfu_vs_peak"] = round(mfu, 2) if mfu else "unknown kind"

print("\n================ REPORT (paste into the Phase 0 log) ================")
print(json.dumps(REPORT, indent=1))
print("""
Decision rules:
- device_kind contains "v5e" and device_count == 8 and hbm_per_chip_GB >= 15 -> Balagh-nano on Kaggle is ON.
- matmul_TFLOPs_per_chip under 100 on v5e -> something is wrong (wrong accelerator, CPU fallback); do not plan on it.
- ckpt_roundtrip_ok False -> fix checkpointing before any multi-session run (orbax to /kaggle/working, then upload to the HF Hub).
- est_tokens_per_week_2B-class x 2 accounts is the honest weekly budget. Next step: run the same probe with the real base model under MaxText.
""")
