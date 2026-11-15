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
per call. Inputs are bfloat16 with a batch of 4. Ragged rows hold all, a half,
a quarter and a sixteenth of the keys; equal rows are all full. One query runs
without a causal mask, 4 and 8 queries with ``mask="causal"``.

Calls that the vector kernels cannot take use the boolean mask on both builds:
more than 32 query rows per key/value head (for example 8 queries with 8 query
heads per key/value head), and head dimension 512.

Memory is the peak above the inputs for a single call, mask construction
included.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time

DEFAULT_SHAPES = (
    # query heads, key/value heads, head dimension
    (32, 8, 128),  # Llama 3.1 8B, Qwen3-8B
    (64, 8, 128),  # Llama 3.1 70B, Qwen3-32B
    (32, 8, 64),  # Llama 3.2 1B
    (16, 8, 256),  # Gemma 2 9B
    (32, 32, 128),  # no grouped queries, Llama 2 7B
    (16, 2, 512),  # head dimension 512: boolean mask on both builds
)
DEFAULT_KEYS = (1024, 2048, 4096, 16384)
QUERIES = (1, 4, 8)
ROWS = ("ragged", "equal")
BATCH = 4
WARMUP = 5


def parse_shape(value):
    try:
        shape = tuple(int(x) for x in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "shape must be q_heads,kv_heads,head_dim"
        ) from error
    if len(shape) != 3 or min(shape) <= 0 or shape[0] % shape[1] != 0:
        raise argparse.ArgumentTypeError(
            "shape must be q_heads,kv_heads,head_dim with q_heads divisible by kv_heads"
        )
    return shape


def run_worker(args):
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
                "label": args.worker,
                "device": info.get("device_name", ""),
                "architecture": info.get("architecture", ""),
                "mlx": mx.__version__,
            }
        ),
        flush=True,
    )

    mx.random.seed(0)
    for q_heads, kv_heads, head_dim in args.shape:
        scale = head_dim**-0.5
        for n in args.keys:
            for rows in ROWS:
                if rows == "ragged":
                    lengths = [n, n // 2, n // 4, n // 16]
                else:
                    lengths = [n] * BATCH
                pads = mx.array([n - length for length in lengths], dtype=mx.int32)
                k = mx.random.normal((BATCH, kv_heads, n, head_dim))
                v = mx.random.normal((BATCH, kv_heads, n, head_dim))
                k, v = k.astype(mx.bfloat16), v.astype(mx.bfloat16)
                for queries in QUERIES:
                    causal = queries > 1
                    q = mx.random.normal((BATCH, q_heads, queries, head_dim))
                    q = q.astype(mx.bfloat16)
                    mx.eval(q, k, v, pads)

                    def step(mode, calls):
                        if mode == "mask":
                            # What a caller passes today: one boolean mask
                            # that every layer shares.
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
                            mx.eval(step(mode, args.chain))
                        samples = []
                        for _ in range(args.rounds):
                            mx.synchronize()
                            start = time.perf_counter()
                            mx.eval(step(mode, args.chain))
                            mx.synchronize()
                            elapsed = time.perf_counter() - start
                            samples.append(elapsed / args.chain * 1e6)

                        outputs[mode] = step(mode, 1)
                        mx.eval(outputs[mode])
                        mx.synchronize()
                        base = mx.get_active_memory()
                        mx.reset_peak_memory()
                        mx.eval(step(mode, 1))
                        mx.synchronize()
                        peak = mx.get_peak_memory() - base

                        record = {
                            "label": args.worker,
                            "mode": mode,
                            "shape": [q_heads, kv_heads, head_dim],
                            "keys": n,
                            "rows": rows,
                            "queries": queries,
                            "us": samples,
                            "peak_bytes": peak,
                        }
                        if mode == "left_padding":
                            diff = outputs["left_padding"] - outputs["mask"]
                            record["max_abs_diff"] = mx.abs(diff).max().item()
                        print(json.dumps(record), flush=True)


def run_driver(args):
    order = (("base", args.base), ("pr", args.pr), ("pr", args.pr), ("base", args.base))
    devices, records = {}, []
    for process, (label, python) in enumerate(order):
        print(f"running {label}: {python}", file=sys.stderr, flush=True)
        command = [python, "-I", __file__, "--worker", label]
        command += ["--rounds", str(args.rounds), "--chain", str(args.chain)]
        command += ["--keys", *map(str, args.keys)]
        for shape in args.shape:
            command += ["--shape", ",".join(map(str, shape))]
        output = subprocess.run(command, check=True, capture_output=True, text=True)
        for line in output.stdout.splitlines():
            record = json.loads(line)
            if "device" in record:
                devices[label] = record
            else:
                record["process"] = process
                records.append(record)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"devices": devices, "records": records}, f)

    samples, process_medians, peaks, diffs = {}, {}, {}, {}
    for r in records:
        case = (tuple(r["shape"]), r["keys"], r["rows"], r["queries"])
        samples.setdefault((r["label"], r["mode"], *case), []).extend(r["us"])
        process_medians.setdefault((r["label"], r["mode"], *case), []).append(
            statistics.median(r["us"])
        )
        peaks.setdefault((r["label"], r["mode"], *case), r["peak_bytes"])
        if "max_abs_diff" in r:
            diffs[case] = max(diffs.get(case, 0.0), r["max_abs_diff"])
    if not diffs:
        sys.exit("--pr build does not accept left_padding")

    pr = devices["pr"]
    print(f"Device: {pr['device']} ({pr['architecture']})")
    print(f"MLX: base {devices['base']['mlx']}, PR {pr['mlx']}")
    print(f"{args.rounds} rounds x 2 processes per build, {args.chain} chained calls")

    def name(queries):
        return f"{queries}, causal" if queries > 1 else "1"

    mib = 1 << 20
    for shape in args.shape:
        q_heads, kv_heads, head_dim = shape
        print()
        print(
            f"### {q_heads} query heads, {kv_heads} key/value heads, "
            f"head dimension {head_dim}"
        )
        print()
        print("Time per attention call, microseconds (median)")
        print()
        print(
            "| Queries | Keys | Rows | Base, mask | PR, mask "
            "| PR, `left_padding` | Change vs base |"
        )
        print("|---|---:|---|---:|---:|---:|---:|")
        for queries in QUERIES:
            for n in args.keys:
                for rows in ROWS:
                    case = (shape, n, rows, queries)
                    base = statistics.median(samples[("base", "mask", *case)])
                    masked = statistics.median(samples[("pr", "mask", *case)])
                    padded = statistics.median(samples[("pr", "left_padding", *case)])
                    print(
                        f"| {name(queries)} | {n:,} | {rows} | {base:.1f} "
                        f"| {masked:.1f} | {padded:.1f} "
                        f"| {100 * (padded / base - 1):+.1f}% |"
                    )
        print()
        print("Peak memory above the inputs for one call, ragged rows, MiB")
        print()
        print("| Queries | Keys | Base, mask | PR, mask | PR, `left_padding` |")
        print("|---|---:|---:|---:|---:|")
        for queries in QUERIES:
            for n in args.keys:
                case = (shape, n, "ragged", queries)
                print(
                    f"| {name(queries)} | {n:,} "
                    f"| {peaks[('base', 'mask', *case)] / mib:.1f} "
                    f"| {peaks[('pr', 'mask', *case)] / mib:.1f} "
                    f"| {peaks[('pr', 'left_padding', *case)] / mib:.1f} |"
                )
    print()
    worst = max(diffs.values())
    print(f"Largest |left_padding - mask| over all cases (bfloat16): {worst:.3g}")

    # The two processes of a build should agree; when they don't, something
    # else ran on the machine and the affected rows should be rerun.
    noisy = []
    for key, medians in process_medians.items():
        spread = max(medians) / min(medians) - 1
        if spread > 0.05:
            label, mode, shape, n, rows, queries = key
            noisy.append(
                f"- {label}, {mode}, shape {','.join(map(str, shape))}, "
                f"{n:,} keys, {rows}, queries {name(queries)}: "
                + " vs ".join(f"{m:.1f}" for m in medians)
                + f" us ({100 * spread:+.0f}%)"
            )
    print()
    if noisy:
        print("Processes of the same build that disagree by more than 5%:")
        print("\n".join(noisy))
    else:
        print("The two processes of each build agree within 5% on every case.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", help="Python of the main-branch environment")
    parser.add_argument("--pr", help="Python of the branch environment")
    parser.add_argument(
        "--shape",
        action="append",
        type=parse_shape,
        help="q_heads,kv_heads,head_dim; repeat for several (default: a set "
        "of common model shapes)",
    )
    parser.add_argument("--keys", type=int, nargs="+", default=list(DEFAULT_KEYS))
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--chain", type=int, default=50)
    parser.add_argument("--out", help="write the raw samples to this JSON file")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.shape = args.shape or list(DEFAULT_SHAPES)
    if args.worker:
        run_worker(args)
    elif args.base and args.pr:
        run_driver(args)
    else:
        parser.error("pass --base and --pr")


if __name__ == "__main__":
    main()
