"""Benchmark ``left_padding`` against a boolean mask in batched SDPA.

Batched decode keeps rows of different lengths in one key/value buffer,
left-padded to the longest row. Today the padding is hidden with a boolean
mask; ``left_padding`` lets the Metal vector kernels skip it instead.

Build main and this branch into separate environments, then run:

    python benchmarks/python/sdpa_left_padding_bench.py \\
        --base /path/to/main-env/bin/python --pr /path/to/branch-env/bin/python

The two builds run in separate processes in the order base, PR, PR, base, and
each timing pools both processes of a build. Each sample chains ``--chain``
attention calls in one evaluation, as stacked layers do, and reports the time
per call. The shape is Qwen3-8B attention (32 query heads, 8 key/value heads,
head dimension 128, bfloat16) with a batch of 4. Ragged rows hold all, a half,
a quarter and a sixteenth of the keys; equal rows are all full. One query
runs without a causal mask, 4 and 8 queries with ``mask="causal"``.

Memory is the peak above the inputs for a single call, mask construction
included.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time

KEYS = (1024, 2048, 4096, 16384)
QUERIES = (1, 4, 8)
ROWS = ("ragged", "equal")
Q_HEADS, KV_HEADS, HEAD_DIM, BATCH = 32, 8, 128, 4
WARMUP = 5


def run_worker(label, rounds, chain):
    import mlx.core as mx

    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    q = mx.zeros((1, 1, 1, 8))
    try:
        mx.fast.scaled_dot_product_attention(
            q, q, q, scale=1.0, left_padding=mx.zeros((1,), mx.int32)
        )
        modes = ("mask", "left_padding")
    except TypeError:
        modes = ("mask",)
    print(
        json.dumps(
            {
                "label": label,
                "device": info.get("device_name", ""),
                "architecture": info.get("architecture", ""),
                "mlx": mx.__version__,
            }
        ),
        flush=True,
    )

    scale = HEAD_DIM**-0.5
    mx.random.seed(0)
    for n in KEYS:
        for rows in ROWS:
            if rows == "ragged":
                lengths = [n, n // 2, n // 4, n // 16]
            else:
                lengths = [n] * BATCH
            pads = mx.array([n - length for length in lengths], dtype=mx.int32)
            k = mx.random.normal((BATCH, KV_HEADS, n, HEAD_DIM)).astype(mx.bfloat16)
            v = mx.random.normal((BATCH, KV_HEADS, n, HEAD_DIM)).astype(mx.bfloat16)
            for queries in QUERIES:
                causal = queries > 1
                q = mx.random.normal((BATCH, Q_HEADS, queries, HEAD_DIM))
                q = q.astype(mx.bfloat16)
                mx.eval(q, k, v, pads)

                def step(mode, calls):
                    if mode == "mask":
                        # What a caller passes today: one boolean mask that
                        # every layer shares.
                        mask = mx.arange(n)[None, :] >= pads[:, None]
                        mask = mask[:, None, None, :]
                        if causal:
                            rows_q = mx.arange(n - queries, n)[:, None]
                            mask = mask & (rows_q >= mx.arange(n)[None, :])
                        kwargs = {"mask": mask}
                    else:
                        kwargs = {
                            "left_padding": pads,
                            "mask": "causal" if causal else None,
                        }
                    x = q
                    for _ in range(calls):
                        x = mx.fast.scaled_dot_product_attention(
                            x, k, v, scale=scale, **kwargs
                        )
                    return x

                outputs = {}
                for mode in modes:
                    for _ in range(WARMUP):
                        mx.eval(step(mode, chain))
                    samples = []
                    for _ in range(rounds):
                        mx.synchronize()
                        start = time.perf_counter()
                        mx.eval(step(mode, chain))
                        mx.synchronize()
                        samples.append((time.perf_counter() - start) / chain * 1e6)

                    outputs[mode] = step(mode, 1)
                    mx.eval(outputs[mode])
                    mx.synchronize()
                    base = mx.get_active_memory()
                    mx.reset_peak_memory()
                    mx.eval(step(mode, 1))
                    mx.synchronize()
                    peak = mx.get_peak_memory() - base

                    record = {
                        "label": label,
                        "mode": mode,
                        "keys": n,
                        "rows": rows,
                        "queries": queries,
                        "us": samples,
                        "peak_bytes": peak,
                    }
                    if mode == "left_padding":
                        diff = mx.abs(outputs["left_padding"] - outputs["mask"])
                        record["max_abs_diff"] = diff.max().item()
                    print(json.dumps(record), flush=True)


def run_driver(args):
    order = (("base", args.base), ("pr", args.pr), ("pr", args.pr), ("base", args.base))
    devices, records = {}, []
    for label, python in order:
        print(f"running {label}: {python}", file=sys.stderr, flush=True)
        command = [python, "-I", __file__, "--worker", label]
        command += ["--rounds", str(args.rounds), "--chain", str(args.chain)]
        output = subprocess.run(command, check=True, capture_output=True, text=True)
        for line in output.stdout.splitlines():
            record = json.loads(line)
            if "device" in record:
                devices[label] = record
            else:
                records.append(record)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"devices": devices, "records": records}, f)

    samples, peaks, diffs = {}, {}, {}
    for r in records:
        key = (r["label"], r["mode"], r["keys"], r["rows"], r["queries"])
        samples.setdefault(key, []).extend(r["us"])
        peaks.setdefault(key, r["peak_bytes"])
        if "max_abs_diff" in r:
            diffs[key] = max(diffs.get(key, 0.0), r["max_abs_diff"])
    if not any(key[1] == "left_padding" for key in samples if key[0] == "pr"):
        sys.exit("--pr build does not accept left_padding")

    pr = devices["pr"]
    print(f"Device: {pr['device']} ({pr['architecture']})")
    print(f"MLX: base {devices['base']['mlx']}, PR {pr['mlx']}")
    print(f"{args.rounds} rounds x 2 processes per build, {args.chain} chained calls")
    print()
    print("Time per attention call, microseconds (median)")
    print()
    print(
        "| Queries | Keys | Rows | Base, mask | PR, mask "
        "| PR, `left_padding` | Change vs base |"
    )
    print("|---|---:|---|---:|---:|---:|---:|")
    for queries in QUERIES:
        for n in KEYS:
            for rows in ROWS:
                case = (n, rows, queries)
                base = statistics.median(samples[("base", "mask", *case)])
                masked = statistics.median(samples[("pr", "mask", *case)])
                padded = statistics.median(samples[("pr", "left_padding", *case)])
                name = f"{queries}, causal" if queries > 1 else "1"
                print(
                    f"| {name} | {n:,} | {rows} | {base:.1f} | {masked:.1f} "
                    f"| {padded:.1f} | {100 * (padded / base - 1):+.1f}% |"
                )
    print()
    print("Peak memory above the inputs for one call, MiB")
    print()
    print("| Queries | Keys | Rows | Base, mask | PR, mask | PR, `left_padding` |")
    print("|---|---:|---|---:|---:|---:|")
    mib = 1 << 20
    for queries in QUERIES:
        for n in KEYS:
            rows = "ragged"
            case = (n, rows, queries)
            name = f"{queries}, causal" if queries > 1 else "1"
            print(
                f"| {name} | {n:,} | {rows} "
                f"| {peaks[('base', 'mask', *case)] / mib:.1f} "
                f"| {peaks[('pr', 'mask', *case)] / mib:.1f} "
                f"| {peaks[('pr', 'left_padding', *case)] / mib:.1f} |"
            )
    print()
    worst = max(diffs.values())
    print(f"Largest |left_padding - mask| over all cases (bfloat16): {worst:.3g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", help="Python of the main-branch environment")
    parser.add_argument("--pr", help="Python of the branch environment")
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--chain", type=int, default=100)
    parser.add_argument("--out", help="write the raw samples to this JSON file")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        run_worker(args.worker, args.rounds, args.chain)
    elif args.base and args.pr:
        run_driver(args)
    else:
        parser.error("pass --base and --pr")


if __name__ == "__main__":
    main()
