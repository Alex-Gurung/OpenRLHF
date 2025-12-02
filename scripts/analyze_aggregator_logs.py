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

import matplotlib.pyplot as plt

BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}", re.IGNORECASE)


def extract_last_boxed(text: str) -> str:
    matches = list(BOXED_RE.finditer(text or ""))
    if matches:
        return matches[-1].group(1).strip()
    return text or ""


def parse_prediction(raw_text: str) -> float:
    candidate = extract_last_boxed(raw_text)
    candidate = (candidate or raw_text or "").strip().lower()
    if "yes" in candidate and "no" not in candidate:
        return 1.0
    return 0.0


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def extract_step(path: Path) -> int:
    # Matches aggregator_step123.jsonl or generator_step123.jsonl
    m = re.search(r"step(\d+)", path.name)
    return int(m.group(1)) if m else -1


def _normalize_group_responses(responses):
    if isinstance(responses, list) and responses and isinstance(responses[0], list):
        return responses[0]
    if responses is None:
        return []
    if isinstance(responses, list):
        return responses
    return [responses]


def analyze_file(path: Path):
    stats = defaultdict(int)
    rewards = []
    positions = []
    agg_preds = []
    mean_elem_preds = []
    agree_fraction = []
    agree_first_pred = []
    agree_majority_pred = []

    for rec in load_jsonl(path):
        agg_resp = rec["aggregator_response"]
        responses = _normalize_group_responses(rec["group_responses"])
        label = rec.get("label")
        agg_reward = rec.get("aggregator_reward", rec.get("reward"))

        agg_pred = parse_prediction(agg_resp)
        elem_preds = [parse_prediction(r) for r in responses]

        stats["total_rows"] += 1

        stats["usable"] += 1
        if agg_resp == responses[0]:
            stats["match_first"] += 1
        pos = next((i for i, r in enumerate(responses) if r == agg_resp), -1)
        if pos >= 0:
            positions.append(pos)
            stats["match_any"] += 1

        mode_resp, _ = Counter(responses).most_common(1)[0]
        if agg_resp == mode_resp:
            stats["match_mode"] += 1

        rewards.append(float(agg_reward))
        agg_preds.append(agg_pred)
        mean_elem_preds.append(sum(elem_preds) / len(elem_preds) if elem_preds else 0.0)
        agree_fraction.append(sum(1 for p in elem_preds if p == agg_pred) / len(elem_preds))
        agree_first_pred.append(1.0 if agg_pred == elem_preds[0] else 0.0)
        majority_pred = 1.0 if sum(elem_preds) >= (len(elem_preds) / 2) else 0.0
        agree_majority_pred.append(1.0 if agg_pred == majority_pred else 0.0)
        if label is not None:
            stats["label_rows"] += 1
            stats["agg_hits_label"] += 1 if agg_pred == float(label) else 0

    total = stats["usable"] or 1
    mean_elem = sum(mean_elem_preds) / len(mean_elem_preds) if mean_elem_preds else 0.0
    mean_agg = sum(agg_preds) / len(agg_preds) if agg_preds else 0.0
    # simple Pearson correlation between agg_pred and mean elem pred
    if agg_preds:
        n = len(agg_preds)
        mean_x = mean_agg
        mean_y = mean_elem
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(agg_preds, mean_elem_preds)) / n
        var_x = sum((x - mean_x) ** 2 for x in agg_preds) / n
        var_y = sum((y - mean_y) ** 2 for y in mean_elem_preds) / n
        corr = cov / ((var_x ** 0.5) * (var_y ** 0.5) + 1e-12)
    else:
        corr = 0.0

    return {
        "count": stats["usable"],
        "rows": stats["total_rows"],
        "match_first_pct": stats["match_first"] / total,
        "match_any_pct": stats["match_any"] / total,
        "match_mode_pct": stats["match_mode"] / total,
        "avg_position": sum(positions) / len(positions) if positions else None,
        "reward_mean": sum(rewards) / len(rewards) if rewards else None,
        "agg_pred_mean": mean_agg,
        "elem_pred_mean": mean_elem,
        "agg_elem_pred_corr": corr,
        "agree_fraction_mean": sum(agree_fraction) / len(agree_fraction) if agree_fraction else 0.0,
        "agree_first_pred_pct": sum(agree_first_pred) / len(agree_first_pred) if agree_first_pred else 0.0,
        "agree_majority_pred_pct": sum(agree_majority_pred) / len(agree_majority_pred) if agree_majority_pred else 0.0,
        "agg_hits_label_pct": stats["agg_hits_label"] / stats["label_rows"] if stats["label_rows"] else None,
    }


def maybe_plot(metrics_by_step, out_dir: Path):
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
    assert paths, f"No aggregator_step*.jsonl found in {args.log_dir}"

    metrics_by_step = {}
    for path in paths:
        step = extract_step(path)
        metrics = analyze_file(path)
        metrics_by_step[step] = metrics

    print(
        "Per-step metrics (n usable/total, reward_mean, agg_pred_mean, elem_pred_mean, "
        "agree_first_pred, agree_majority_pred, agg_elem_pred_corr):"
    )
    for step in sorted(metrics_by_step.keys()):
        m = metrics_by_step[step]
        base = (
            f"{step:>6}: n={m['count']:>4}/{m['rows']:>4}, "
            f"reward={m['reward_mean']:.3f}, "
            f"agg_pred={m['agg_pred_mean']:.3f}, elems_pred={m['elem_pred_mean']:.3f}, "
            f"agree_first_pred={m['agree_first_pred_pct']*100:5.1f}%, "
            f"agree_majority_pred={m['agree_majority_pred_pct']*100:5.1f}%, "
            f"corr={m['agg_elem_pred_corr']:.3f}"
        )
        print(base)

    if args.plot:
        args.log_dir.mkdir(parents=True, exist_ok=True)
        maybe_plot(metrics_by_step, args.log_dir)
        print(f"Plots written to {args.log_dir}")


if __name__ == "__main__":
    main()
