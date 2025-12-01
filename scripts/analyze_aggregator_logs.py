"""Post-train aggregator analysis.

Reads JSONL sample artifacts written by `ppo_trainer` (e.g.,
`sample_logs/aggregator_stepXXXX.jsonl`) and computes quick diagnostics:

- How often the aggregator chose the first generator response
- How often it matched any generator response (sanity check)
- How often it picked the mode/majority response
- Distribution over which position it selected
- Average aggregator reward

Optionally writes simple Matplotlib visualizations over training steps.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def extract_step(path: Path) -> int:
    # Matches aggregator_step123.jsonl or generator_step123.jsonl
    m = re.search(r"step(\d+)", path.name)
    return int(m.group(1)) if m else -1


def analyze_file(path: Path):
    stats = defaultdict(int)
    rewards = []
    positions = []

    for rec in load_jsonl(path):
        agg_resp = rec.get("aggregator_response") or rec.get("response")
        responses = rec.get("group_responses")
        if isinstance(responses, list) and responses and isinstance(responses[0], list):
            # unwrap if nested
            responses = responses[0]

        if not agg_resp or not responses:
            continue

        stats["total"] += 1
        if agg_resp == responses[0]:
            stats["match_first"] += 1
        # position of first match (if any)
        try:
            pos = responses.index(agg_resp)
            positions.append(pos)
            stats["match_any"] += 1
        except ValueError:
            pos = None

        if responses:
            mode_resp, _ = Counter(responses).most_common(1)[0]
            if agg_resp == mode_resp:
                stats["match_mode"] += 1

        rew = rec.get("aggregator_reward")
        if rew is None:
            rew = rec.get("reward")
        if rew is not None:
            try:
                rewards.append(float(rew))
            except Exception:
                pass

    total = stats["total"] or 1
    return {
        "count": stats["total"],
        "match_first_pct": stats["match_first"] / total,
        "match_any_pct": stats["match_any"] / total,
        "match_mode_pct": stats["match_mode"] / total,
        "avg_position": sum(positions) / len(positions) if positions else None,
        "reward_mean": sum(rewards) / len(rewards) if rewards else None,
    }


def maybe_plot(metrics_by_step, out_dir: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping plots")
        return

    steps = sorted(metrics_by_step.keys())
    if not steps:
        return

    def plot_series(metric_key, ylabel, filename):
        vals = [metrics_by_step[s].get(metric_key) for s in steps]
        plt.figure(figsize=(6, 3))
        plt.plot(steps, vals, marker="o")
        plt.xlabel("step")
        plt.ylabel(ylabel)
        plt.title(metric_key)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / filename)
        plt.close()

    plot_series("match_first_pct", "P(agg==first)", "match_first_pct.png")
    plot_series("match_mode_pct", "P(agg==mode)", "match_mode_pct.png")
    plot_series("reward_mean", "Aggregator reward", "aggregator_reward_mean.png")


def main():
    parser = argparse.ArgumentParser(description="Analyze aggregator sample artifacts")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("sample_logs"),
        help="Directory containing aggregator_step*.jsonl",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save matplotlib plots in the log dir",
    )
    args = parser.parse_args()

    paths = sorted(args.log_dir.glob("aggregator_step*.jsonl"), key=extract_step)
    if not paths:
        print(f"No aggregator_step*.jsonl found in {args.log_dir}")
        return

    metrics_by_step = {}
    for path in paths:
        step = extract_step(path)
        metrics = analyze_file(path)
        metrics_by_step[step] = metrics

    print("Per-step metrics (step: count, match_first, match_mode, reward_mean):")
    for step in sorted(metrics_by_step.keys()):
        m = metrics_by_step[step]
        base = f"{step:>6}: n={m['count']:>4}, first={m['match_first_pct']*100:5.1f}%, mode={m['match_mode_pct']*100:5.1f}%"
        if m.get("reward_mean") is not None:
            base += f", reward={m['reward_mean']:.3f}"
        print(base)

    if args.plot:
        args.log_dir.mkdir(parents=True, exist_ok=True)
        maybe_plot(metrics_by_step, args.log_dir)
        print(f"Plots written to {args.log_dir}")


if __name__ == "__main__":
    main()
