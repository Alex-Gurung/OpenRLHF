"""Quota-balanced group sampling for MC estimation.

Each solution appears in exactly `quota` groups for unbiased MC estimates.
Total groups G = (n * quota) / k
"""

import random
from typing import Dict, List, Optional


def quota_sample_groups(
    n: int,
    k: int,
    quota: int,
    seed: Optional[int] = None,
) -> List[List[int]]:
    """Sample groups where each solution appears in EXACTLY `quota` groups.

    Each group contains k DISTINCT elements. The algorithm uses greedy assignment
    with backtracking-free selection to achieve exact quota while avoiding duplicates.

    Args:
        n: Total number of solutions
        k: Group size (solutions per group, must have k distinct elements)
        quota: Target appearances per solution (each solution in exactly quota groups)
        seed: Random seed for reproducibility

    Returns:
        List of G groups, each group is a sorted list of k DISTINCT solution indices.
        G = (n * quota) / k (must be integer)

    Raises:
        ValueError: If n*quota is not divisible by k (exact quota impossible)

    Example:
        >>> groups = quota_sample_groups(n=8, k=4, quota=4)
        >>> len(groups)  # G = 8*4/4 = 8 groups
        8
        >>> coverage = get_element_coverage(groups, n=8)
        >>> all(c == 4 for c in coverage["in_counts"])  # Each element in exactly 4 groups
        True
        >>> all(len(g) == len(set(g)) for g in groups)  # All groups have distinct elements
        True
    """
    if k > n:
        raise ValueError(f"group_size k={k} cannot exceed n={n} solutions")
    if quota < 1:
        raise ValueError(f"quota must be >= 1, got {quota}")
    if (n * quota) % k != 0:
        raise ValueError(
            f"n*quota ({n * quota}) must be divisible by k ({k}) for exact quota guarantee. "
            f"Try adjusting quota or group_size."
        )

    if seed is not None:
        random.seed(seed)

    # Number of groups
    g = (n * quota) // k

    # Track remaining quota for each element
    remaining = {i: quota for i in range(n)}

    groups = []
    for _ in range(g):
        # Get elements that still need to appear, grouped by remaining quota
        # Shuffle within each quota level for randomness
        quota_to_elems: Dict[int, List[int]] = {}
        for i in range(n):
            if remaining[i] > 0:
                if remaining[i] not in quota_to_elems:
                    quota_to_elems[remaining[i]] = []
                quota_to_elems[remaining[i]].append(i)

        # Shuffle within each quota level
        for elems in quota_to_elems.values():
            random.shuffle(elems)

        # Build available list: highest quota first, shuffled within quota levels
        available = []
        for q in sorted(quota_to_elems.keys(), reverse=True):
            available.extend(quota_to_elems[q])

        # Select k distinct elements with highest remaining quotas
        group = []
        for elem in available:
            if len(group) >= k:
                break
            group.append(elem)
            remaining[elem] -= 1

        if len(group) < k:
            # This shouldn't happen if n*quota is divisible by k and quota <= n-k+1
            raise RuntimeError(
                f"Could not form valid group: only {len(group)} elements available, need {k}. "
                f"This can happen if quota > n - k + 1."
            )

        groups.append(sorted(group))

    return groups


def get_element_coverage(groups: List[List[int]], n: int) -> Dict[str, List[int]]:
    """Get coverage statistics for each element.

    Args:
        groups: List of groups (each group is list of element indices)
        n: Total number of elements

    Returns:
        Dict with:
            "in_counts": List of how many groups each element appears in
            "out_counts": List of how many groups each element is excluded from
    """
    in_counts = [0] * n
    total_groups = len(groups)

    for group in groups:
        group_set = set(group)
        for i in range(n):
            if i in group_set:
                in_counts[i] += 1

    out_counts = [total_groups - c for c in in_counts]

    return {
        "in_counts": in_counts,
        "out_counts": out_counts,
    }


def verify_quota_balance(groups: List[List[int]], n: int, quota: int, tolerance: int = 1) -> bool:
    """Verify that quota sampling achieved balanced coverage.

    Args:
        groups: Sampled groups
        n: Number of elements
        quota: Target quota
        tolerance: Allowed deviation from quota

    Returns:
        True if all elements appear in quota ± tolerance groups
    """
    coverage = get_element_coverage(groups, n)
    for count in coverage["in_counts"]:
        if abs(count - quota) > tolerance:
            return False
    return True
