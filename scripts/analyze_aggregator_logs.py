"""Post-train aggregator analysis.

Reads JSONL sample artifacts written by `ppo_trainer` (e.g.,
`sample_logs/aggregator_stepXXXX.jsonl`) and computes quick diagnostics:

- How often the aggregator chose the first generator response
- How often it matched any generator response (sanity check)
- How often it picked the mode/majority response
- Distribution over which position it selected
- Average aggregator reward

Metrics reported per step:
- Reward: mean aggregator reward.
- Agg Acc: percent of examples where aggregator prediction matches label.
- Maj Acc: percent where majority vote of group predictions matches label.
- First Acc: percent where first response prediction matches label.
- Agg Pred / Elem Pred: average yes/no predictions (1=yes) for aggregator and group elements.
- Agree First / Agree Maj: how often aggregator prediction matches first / majority prediction.
- Corr: Pearson correlation between aggregator prediction and mean group prediction.

Plots are saved if --plot is provided.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

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
    agg_correct = []
    first_correct = []
    majority_correct = []

    for rec in load_jsonl(path):
        agg_resp = rec["aggregator_response"]
        responses = _normalize_group_responses(rec.get("group_responses"))
        label = rec.get("label")
        agg_reward = rec.get("aggregator_reward", rec.get("reward"))

        agg_pred = parse_prediction(agg_resp)
        elem_preds = [parse_prediction(r) for r in responses] if responses else []

        stats["total_rows"] += 1

        has_group = bool(responses)
        if has_group:
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
        if elem_preds:
            mean_elem_preds.append(sum(elem_preds) / len(elem_preds))
            agree_fraction.append(sum(1 for p in elem_preds if p == agg_pred) / len(elem_preds))
            agree_first_pred.append(1.0 if agg_pred == elem_preds[0] else 0.0)
            majority_pred = 1.0 if sum(elem_preds) >= (len(elem_preds) / 2) else 0.0
            agree_majority_pred.append(1.0 if agg_pred == majority_pred else 0.0)
        if label is not None:
            stats["label_rows"] += 1
            stats["agg_hits_label"] += 1 if agg_pred == float(label) else 0
            if elem_preds:
                first_correct.append(1.0 if elem_preds[0] == float(label) else 0.0)
                majority_pred = 1.0 if sum(elem_preds) >= (len(elem_preds) / 2) else 0.0
                majority_correct.append(1.0 if majority_pred == float(label) else 0.0)
            agg_correct.append(1.0 if agg_pred == float(label) else 0.0)

    total = stats["usable"] or 1
    mean_elem = sum(mean_elem_preds) / len(mean_elem_preds) if mean_elem_preds else None
    mean_agg = sum(agg_preds) / len(agg_preds) if agg_preds else None
    # simple Pearson correlation between agg_pred and mean elem pred when both exist
    if agg_preds and mean_elem_preds:
        n = len(mean_elem_preds)
        mean_x = mean_agg
        mean_y = mean_elem
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(agg_preds[:n], mean_elem_preds)) / n
        var_x = sum((x - mean_x) ** 2 for x in agg_preds[:n]) / n
        var_y = sum((y - mean_y) ** 2 for y in mean_elem_preds) / n
        corr = cov / ((var_x ** 0.5) * (var_y ** 0.5) + 1e-12)
    else:
        corr = None

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
        "agree_fraction_mean": sum(agree_fraction) / len(agree_fraction) if agree_fraction else None,
        "agree_first_pred_pct": sum(agree_first_pred) / len(agree_first_pred) if agree_first_pred else None,
        "agree_majority_pred_pct": sum(agree_majority_pred) / len(agree_majority_pred) if agree_majority_pred else None,
        "agg_hits_label_pct": stats["agg_hits_label"] / stats["label_rows"] if stats["label_rows"] else None,
        "first_hits_label_pct": sum(first_correct) / len(first_correct) if first_correct else None,
        "majority_hits_label_pct": sum(majority_correct) / len(majority_correct) if majority_correct else None,
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

    plot_series("reward_mean", "Aggregator reward", "aggregator_reward_mean.png")
    plot_series("agg_hits_label_pct", "Aggregator accuracy", "agg_accuracy.png")
    plot_series("majority_hits_label_pct", "Majority accuracy", "majority_accuracy.png")
    plot_series("agree_first_pred_pct", "Agree with first pred", "agree_first_pred.png")
    plot_series("agree_majority_pred_pct", "Agree with majority pred", "agree_majority_pred.png")


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
    parser.add_argument(
        "--pattern",
        type=str,
        default="aggregator_step*.jsonl",
        help="Filename glob to read (e.g., aggregator_step*.jsonl or generator_step*.jsonl)",
    )
    args = parser.parse_args()

    paths = sorted(args.log_dir.glob(args.pattern), key=extract_step)
    assert paths, f"No files matching {args.pattern} found in {args.log_dir}"

    metrics_by_step = {}
    for path in paths:
        step = extract_step(path)
        metrics = analyze_file(path)
        metrics_by_step[step] = metrics

    console = Console()
    table = Table(title="Aggregator metrics per step", show_lines=False, caption="Yes/no predictions parsed via boxed{...}")
    table.add_column("Step", justify="right")
    table.add_column("n (usable/total)", justify="right")
    table.add_column("Reward", justify="right")
    table.add_column("Agg Acc", justify="right")
    table.add_column("Maj Acc", justify="right")
    table.add_column("First Acc", justify="right")
    table.add_column("Agg Pred", justify="right")
    table.add_column("Elem Pred", justify="right")
    table.add_column("Agree First", justify="right")
    table.add_column("Agree Maj", justify="right")
    table.add_column("Corr", justify="right")

    for step in sorted(metrics_by_step.keys()):
        m = metrics_by_step[step]
        table.add_row(
            str(step),
            f"{m['count']}/{m['rows']}",
            f"{m['reward_mean']:.3f}" if m.get("reward_mean") is not None else "NA",
            f"{m['agg_hits_label_pct']*100:5.1f}%" if m.get("agg_hits_label_pct") is not None else "NA",
            f"{m['majority_hits_label_pct']*100:5.1f}%" if m.get("majority_hits_label_pct") is not None else "NA",
            f"{m['first_hits_label_pct']*100:5.1f}%" if m.get("first_hits_label_pct") is not None else "NA",
            f"{m['agg_pred_mean']:.3f}" if m.get("agg_pred_mean") is not None else "NA",
            f"{m['elem_pred_mean']:.3f}" if m.get("elem_pred_mean") is not None else "NA",
            f"{m['agree_first_pred_pct']*100:5.1f}%" if m.get("agree_first_pred_pct") is not None else "NA",
            f"{m['agree_majority_pred_pct']*100:5.1f}%" if m.get("agree_majority_pred_pct") is not None else "NA",
            f"{m['agg_elem_pred_corr']:.3f}" if m.get("agg_elem_pred_corr") is not None else "NA",
        )

    console.print(table)

    if args.plot:
        args.log_dir.mkdir(parents=True, exist_ok=True)
        maybe_plot(metrics_by_step, args.log_dir)
        print(f"Plots written to {args.log_dir}")


if __name__ == "__main__":
    main()
