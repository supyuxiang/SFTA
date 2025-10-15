import re

# Parse self-evaluation score from model output.
# Expected to find a line like: "SCORE: d" where d in 0..9

def parse_self_score(text: str) -> int:
    m = re.search(r"SCORE:\s*([0-9])\b", text.strip())
    if not m:
        return 0
    return int(m.group(1))


def my_reward_fn(data_source, solution_str, ground_truth, extra_info=None):
    # Use the self-reported score directly as reward (float 0..9)
    score = parse_self_score(solution_str)
    return float(score)


