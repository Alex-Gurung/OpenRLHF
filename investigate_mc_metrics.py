#!/usr/bin/env python3
"""Investigate MC training metrics from wandb."""

import wandb

api = wandb.Api()
run = api.run("alexgurung/livecode_rl/runs/saqu6rez")

# Get history
history = run.history()

print("=" * 60)
print("Available columns:")
print("=" * 60)
for col in sorted(history.columns):
    if 'train' in col.lower() or col.startswith('_'):
        print(f"  {col}")

print("\n" + "=" * 60)
print("MC Metrics Analysis")
print("=" * 60)

# MC mean - should this be zero?
if 'train/mc/mean' in history.columns:
    mc_mean = history['train/mc/mean'].dropna()
    print(f"\ntrain/mc/mean:")
    print(f"  count: {len(mc_mean)}")
    print(f"  min: {mc_mean.min()}")
    print(f"  max: {mc_mean.max()}")
    print(f"  mean: {mc_mean.mean()}")
    print(f"  values: {mc_mean.tolist()[:10]}...")

# MC std
if 'train/mc/std' in history.columns:
    mc_std = history['train/mc/std'].dropna()
    print(f"\ntrain/mc/std:")
    print(f"  count: {len(mc_std)}")
    print(f"  min: {mc_std.min()}")
    print(f"  max: {mc_std.max()}")
    print(f"  mean: {mc_std.mean()}")

print("\n" + "=" * 60)
print("Generator Metrics")
print("=" * 60)

if 'train/gen/pass_rate' in history.columns:
    gen_pass = history['train/gen/pass_rate'].dropna()
    print(f"\ntrain/gen/pass_rate:")
    print(f"  count: {len(gen_pass)}")
    print(f"  min: {gen_pass.min():.4f}")
    print(f"  max: {gen_pass.max():.4f}")
    print(f"  mean: {gen_pass.mean():.4f}")

if 'train/gen/reward_mean' in history.columns:
    gen_reward = history['train/gen/reward_mean'].dropna()
    print(f"\ntrain/gen/reward_mean:")
    print(f"  count: {len(gen_reward)}")
    print(f"  min: {gen_reward.min():.4f}")
    print(f"  max: {gen_reward.max():.4f}")
    print(f"  mean: {gen_reward.mean():.4f}")

print("\n" + "=" * 60)
print("Aggregator Metrics")
print("=" * 60)

if 'train/agg/pass_rate' in history.columns:
    agg_pass = history['train/agg/pass_rate'].dropna()
    print(f"\ntrain/agg/pass_rate:")
    print(f"  count: {len(agg_pass)}")
    print(f"  min: {agg_pass.min():.4f}")
    print(f"  max: {agg_pass.max():.4f}")
    print(f"  mean: {agg_pass.mean():.4f}")

if 'train/agg/lift' in history.columns:
    agg_lift = history['train/agg/lift'].dropna()
    print(f"\ntrain/agg/lift:")
    print(f"  count: {len(agg_lift)}")
    print(f"  min: {agg_lift.min():.4f}")
    print(f"  max: {agg_lift.max():.4f}")
    print(f"  mean: {agg_lift.mean():.4f}")

if 'train/agg/group_reward_mean' in history.columns:
    grp_reward = history['train/agg/group_reward_mean'].dropna()
    print(f"\ntrain/agg/group_reward_mean:")
    print(f"  count: {len(grp_reward)}")
    print(f"  min: {grp_reward.min():.4f}")
    print(f"  max: {grp_reward.max():.4f}")
    print(f"  mean: {grp_reward.mean():.4f}")

# Pass@k metrics
print("\n" + "=" * 60)
print("Pass@k Metrics")
print("=" * 60)

for k in [1, 2, 4, 8]:
    col = f'train/agg/pass@{k}'
    if col in history.columns:
        vals = history[col].dropna()
        print(f"\n{col}:")
        print(f"  count: {len(vals)}")
        print(f"  min: {vals.min():.4f}")
        print(f"  max: {vals.max():.4f}")
        print(f"  mean: {vals.mean():.4f}")
        print(f"  std: {vals.std():.4f}")

print("\n" + "=" * 60)
print("Filter Metrics")
print("=" * 60)

if 'train/filter/problem_keep_rate' in history.columns:
    keep_rate = history['train/filter/problem_keep_rate'].dropna()
    print(f"\ntrain/filter/problem_keep_rate:")
    print(f"  count: {len(keep_rate)}")
    print(f"  min: {keep_rate.min():.4f}")
    print(f"  max: {keep_rate.max():.4f}")
    print(f"  mean: {keep_rate.mean():.4f}")

print("\n" + "=" * 60)
print("Correlation Analysis")
print("=" * 60)

# Check if gen pass rate and agg lift are correlated
if 'train/gen/pass_rate' in history.columns and 'train/agg/lift' in history.columns:
    df = history[['train/gen/pass_rate', 'train/agg/lift', 'train/agg/pass_rate']].dropna()
    if len(df) > 5:
        print(f"\nCorrelation matrix:")
        print(df.corr().to_string())

print("\n" + "=" * 60)
print("Step-by-step values (first 20 steps)")
print("=" * 60)

cols_of_interest = [
    'train/global_step',
    'train/gen/pass_rate',
    'train/agg/pass_rate',
    'train/agg/lift',
    'train/mc/mean',
    'train/filter/problem_keep_rate'
]
available_cols = [c for c in cols_of_interest if c in history.columns]
if available_cols:
    print(history[available_cols].head(20).to_string())

print("\n" + "=" * 60)
print("Aggregator Improvement Over Time")
print("=" * 60)

# Check if agg/pass_rate trends upward
if 'train/agg/pass_rate' in history.columns and 'train/global_step' in history.columns:
    df = history[['train/global_step', 'train/agg/pass_rate', 'train/gen/pass_rate', 'train/agg/lift']].dropna()
    if len(df) > 10:
        # Split into first half and second half
        mid = len(df) // 2
        first_half = df.iloc[:mid]
        second_half = df.iloc[mid:]

        print(f"\nFirst half (steps {first_half['train/global_step'].min():.0f}-{first_half['train/global_step'].max():.0f}):")
        print(f"  agg/pass_rate mean: {first_half['train/agg/pass_rate'].mean():.4f}")
        print(f"  gen/pass_rate mean: {first_half['train/gen/pass_rate'].mean():.4f}")
        print(f"  agg/lift mean: {first_half['train/agg/lift'].mean():.4f}")

        print(f"\nSecond half (steps {second_half['train/global_step'].min():.0f}-{second_half['train/global_step'].max():.0f}):")
        print(f"  agg/pass_rate mean: {second_half['train/agg/pass_rate'].mean():.4f}")
        print(f"  gen/pass_rate mean: {second_half['train/gen/pass_rate'].mean():.4f}")
        print(f"  agg/lift mean: {second_half['train/agg/lift'].mean():.4f}")

        print(f"\nChange (second - first):")
        print(f"  agg/pass_rate: {second_half['train/agg/pass_rate'].mean() - first_half['train/agg/pass_rate'].mean():+.4f}")
        print(f"  gen/pass_rate: {second_half['train/gen/pass_rate'].mean() - first_half['train/gen/pass_rate'].mean():+.4f}")
        print(f"  agg/lift: {second_half['train/agg/lift'].mean() - first_half['train/agg/lift'].mean():+.4f}")

        # Linear regression to check trend
        from scipy import stats
        steps = df['train/global_step'].values
        agg_pass = df['train/agg/pass_rate'].values
        gen_pass = df['train/gen/pass_rate'].values
        lift = df['train/agg/lift'].values

        slope_agg, _, r_agg, p_agg, _ = stats.linregress(steps, agg_pass)
        slope_gen, _, r_gen, p_gen, _ = stats.linregress(steps, gen_pass)
        slope_lift, _, r_lift, p_lift, _ = stats.linregress(steps, lift)

        print(f"\nLinear trend analysis:")
        print(f"  agg/pass_rate: slope={slope_agg:.6f}/step, R²={r_agg**2:.4f}, p={p_agg:.4f}")
        print(f"  gen/pass_rate: slope={slope_gen:.6f}/step, R²={r_gen**2:.4f}, p={p_gen:.4f}")
        print(f"  agg/lift: slope={slope_lift:.6f}/step, R²={r_lift**2:.4f}, p={p_lift:.4f}")
