"""Example reward function for code generation tasks.

This module provides a task-specific reward function that:
1. Extracts code from aggregator outputs
2. Executes code against test cases
3. Returns pass/fail rewards

Users should customize this for their specific code execution setup.
"""

import re
import subprocess
import tempfile
import os
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import json


def extract_code(text: str, language: str = "python") -> Optional[str]:
    """Extract code from markdown code blocks or raw text.

    Args:
        text: The full response text
        language: Expected language (python, cpp, etc.)

    Returns:
        Extracted code string, or None if no code found
    """
    # Try to find code in markdown blocks
    patterns = [
        rf"```{language}\n(.*?)```",
        rf"```{language.lower()}\n(.*?)```",
        r"```\n(.*?)```",
        rf"<code>\n?(.*?)</code>",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            # Return the last match (often the final answer)
            return matches[-1].strip()

    # If no code blocks, check if the whole thing looks like code
    lines = text.strip().split('\n')
    code_indicators = ['def ', 'class ', 'import ', 'from ', 'if ', 'for ', 'while ', 'return ']
    if any(any(line.strip().startswith(ind) for ind in code_indicators) for line in lines):
        return text.strip()

    return None


def execute_python_code(
    code: str,
    test_cases: List[Dict[str, Any]],
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """Execute Python code against test cases.

    Args:
        code: Python code to execute
        test_cases: List of dicts with 'input' and 'expected_output' keys
        timeout: Maximum execution time in seconds

    Returns:
        Dict with 'passed', 'total', 'pass_rate', and 'details'
    """
    results = []

    for i, test in enumerate(test_cases):
        test_input = test.get('input', '')
        expected = test.get('expected_output', '')

        # Create a temporary file with the code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            # Run the code with input
            result = subprocess.run(
                ['python', temp_path],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            actual_output = result.stdout.strip()
            expected_output = expected.strip()

            # Check if output matches
            passed = actual_output == expected_output

            results.append({
                'test_idx': i,
                'passed': passed,
                'expected': expected_output,
                'actual': actual_output,
                'stderr': result.stderr,
            })

        except subprocess.TimeoutExpired:
            results.append({
                'test_idx': i,
                'passed': False,
                'error': 'timeout',
            })
        except Exception as e:
            results.append({
                'test_idx': i,
                'passed': False,
                'error': str(e),
            })
        finally:
            os.unlink(temp_path)

    n_passed = sum(1 for r in results if r['passed'])
    return {
        'passed': n_passed,
        'total': len(test_cases),
        'pass_rate': n_passed / len(test_cases) if test_cases else 0.0,
        'all_passed': n_passed == len(test_cases),
        'details': results,
    }


def load_test_cases(label: Any) -> List[Dict[str, Any]]:
    """Load test cases from label.

    Override this function to load test cases from your data format.
    The label could be:
    - A dict with 'test_cases' key
    - A path to a JSON file
    - A problem ID to look up
    - etc.

    Args:
        label: The label from the dataset (format depends on your data)

    Returns:
        List of test case dicts with 'input' and 'expected_output' keys
    """
    if label is None:
        return []

    if isinstance(label, dict):
        return label.get('test_cases', [])

    if isinstance(label, str):
        # Could be a JSON string or file path
        try:
            data = json.loads(label)
            if isinstance(data, list):
                return data
            return data.get('test_cases', [])
        except json.JSONDecodeError:
            pass

        # Could be a file path
        if os.path.exists(label):
            with open(label) as f:
                data = json.load(f)
                return data.get('test_cases', [])

    return []


def reward_func(
    queries: List[str],
    prompts: List[str],
    labels: List[Any],
    **kwargs,
) -> Dict[str, Any]:
    """Reward function for code generation tasks.

    This function:
    1. Extracts code from each query (aggregator output)
    2. Executes against test cases from labels
    3. Returns binary pass/fail rewards

    Args:
        queries: Aggregator outputs (full text including prompt+response)
        prompts: The aggregation prompts that were given to the aggregator
        labels: Labels containing test case information
        **kwargs: Additional arguments (e.g., 'timeout', 'language')

    Returns:
        Dict with 'rewards' (for advantage), 'scores' (for filtering), 'extra_logs'
    """
    timeout = kwargs.get('timeout', 10.0)
    language = kwargs.get('language', 'python')

    rewards = []
    scores = []
    details = []

    for query, prompt, label in zip(queries, prompts, labels):
        # Extract response from query (remove prompt prefix if present)
        response = query
        if prompt and query.startswith(prompt):
            response = query[len(prompt):]

        # Extract code
        code = extract_code(response, language)

        if code is None:
            rewards.append(-1.0)  # Penalty for no code
            scores.append(0.0)
            details.append({'error': 'no_code_found'})
            continue

        # Load test cases
        test_cases = load_test_cases(label)

        if not test_cases:
            # No test cases - can't evaluate
            rewards.append(0.0)
            scores.append(0.5)
            details.append({'error': 'no_test_cases'})
            continue

        # Execute code
        result = execute_python_code(code, test_cases, timeout)

        # Binary reward: +1 if all pass, -1 otherwise
        reward = 1.0 if result['all_passed'] else -1.0
        score = result['pass_rate']

        rewards.append(reward)
        scores.append(score)
        details.append(result)

    return {
        'rewards': rewards,
        'scores': scores,
        'extra_logs': {'execution_details': details},
    }


# For use with MC aggregation
def mc_reward_func(
    prompts: List[str],
    outputs: List[str],
    labels: List[Any],
    **kwargs,
) -> List[float]:
    """Simplified reward function interface for MC aggregation.

    Args:
        prompts: Aggregation prompts
        outputs: Aggregator outputs (responses only, not full query)
        labels: Labels with test case information

    Returns:
        List of reward floats (one per output)
    """
    timeout = kwargs.get('timeout', 10.0)
    language = kwargs.get('language', 'python')

    rewards = []

    for output, label in zip(outputs, labels):
        code = extract_code(output, language)

        if code is None:
            rewards.append(-1.0)
            continue

        test_cases = load_test_cases(label)

        if not test_cases:
            rewards.append(0.0)
            continue

        result = execute_python_code(code, test_cases, timeout)

        # Binary reward
        reward = 1.0 if result['all_passed'] else -1.0
        rewards.append(reward)

    return rewards


# Async version for better throughput
def mc_reward_func_async(
    prompts: List[str],
    outputs: List[str],
    labels: List[Any],
    max_workers: int = 16,
    **kwargs,
) -> List[float]:
    """Async version of mc_reward_func using thread pool.

    Useful when executing many code samples in parallel.
    """
    timeout = kwargs.get('timeout', 10.0)
    language = kwargs.get('language', 'python')

    def evaluate_single(args):
        output, label = args
        code = extract_code(output, language)

        if code is None:
            return -1.0

        test_cases = load_test_cases(label)

        if not test_cases:
            return 0.0

        result = execute_python_code(code, test_cases, timeout)
        return 1.0 if result['all_passed'] else -1.0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        rewards = list(executor.map(evaluate_single, zip(outputs, labels)))

    return rewards
