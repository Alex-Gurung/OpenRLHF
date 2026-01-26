# Evaluation Script Issue Analysis

## The Problem

BaseAgg is suspiciously high at step 8 for both MC and standard RL training.

## Code Flow Analysis

### eval_checkpoints_mc.py (Checkpoint Evaluation)

1. **Line 567**: Generates solutions with CHECKPOINT model
   ```python
   outputs, codes = await generate_solutions(client, model, problem, n, ...)
   ```

2. **Line 579-591**: Aggregates with CHECKPOINT model
   ```python
   agg_task = run_aggregations(client, model, problem, outputs, subsets, ...)
   ```

3. **Line 600**: Computes MC from CHECKPOINT aggregations
   ```python
   mc_values = compute_marginal_contributions(list(agg_results.items()), n)
   ```

4. **Line 603**: Computes random_pass BEFORE adding high/low MC subsets
   ```python
   # Compute random baseline BEFORE adding extra subsets
   random_pass = float(np.mean(list(agg_results.values()))) if agg_results else 0.0
   ```

5. **Line 614-643**: Adds high/low MC subsets to agg_results (AFTER computing random_pass)

### eval_base_aggregator.py (Base Aggregator Evaluation)

1. **Line 126**: Uses checkpoint solutions from saved generations
   ```python
   outputs = gen_data["outputs"]  # From checkpoint generations
   ```

2. **Line 141-152**: Aggregates with BASE model
   ```python
   agg_results = await run_aggregations(client, model, problem, outputs, subsets, ...)
   # model is BASE model here
   ```

3. **Line 155**: Computes MC from BASE aggregations (DIFFERENT from checkpoint!)
   ```python
   mc = compute_marginal_contributions(list(agg_results.items()), n)
   ```

4. **Line 158**: Computes random_agg_pass from ALL subsets in agg_results
   ```python
   random_agg_pass = sum(agg_results.values()) / len(agg_results) if agg_results else 0.0
   ```

## The Issue

**Key Difference**: In `eval_base_aggregator.py`, `random_agg_pass` is computed from ALL subsets in `agg_results`, which may include high/low MC subsets if they were already computed in the original evaluation.

But more importantly: **The MC values are computed from BASE model aggregations, not checkpoint aggregations!**

This means:
- High/low MC subsets identified in base aggregator are DIFFERENT from checkpoint evaluation
- The MC values reflect BASE model's view of which solutions are good/bad
- This could lead to different high/low MC subsets being aggregated

## Potential Bug

**Question**: Does `random_agg_pass` in base aggregator include high/low MC subsets?

If the generations file includes high/low MC subsets in the saved `subsets`, then:
- Base aggregator will aggregate those subsets
- `random_agg_pass` will include them in the mean
- This could inflate BaseAgg if high MC subsets are included

## How to Verify

1. Check what subsets are in the saved generations files
2. Check if high/low MC subsets are included in the saved subsets
3. Compare the subsets used in checkpoint vs base aggregator evaluation
4. Check if random_agg_pass computation is consistent between the two scripts

## Recommendation

The issue might be that:
- Base aggregator is using a different set of subsets than checkpoint evaluation
- Or high/low MC subsets are being included in random_agg_pass calculation
- Or there's a mismatch in how random_agg_pass is computed

Need to verify the actual subsets being used in both evaluations.
