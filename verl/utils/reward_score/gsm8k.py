# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

_SOLUTION_CLIP_CHARS = 3000


def extract_solution(solution_str, method="strict"):
    global re  # 明确声明使用全局的re模块
    assert method in ["strict", "flexible"]

    # Optimization: Regular expression matching on very long strings can be slow.
    # For math problems, the final answer is usually at the end.
    # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    print("--------------------------------")
    print(f'solution_str_extract: {solution_str}')
    print("--------------------------------")

    if method == "strict":
        final_answer = None
        # 匹配各种答案格式变体
        solution_variants = [
            r"####\s*(\-?[0-9\.\,]+)",                     # ####18
            r"\\\(\\boxed\{\s*(\-?[0-9\.\,]+)\s*\\\)\}",   # \(\boxed{18}\)
            r"\\boxed\{\s*(\-?[0-9\.\,]+)\s*\}",           # \boxed{18}
            r"\\box\s*\{\s*(\-?[0-9\.\,]+)\s*\}",          # \box{18}
            r"\\answer\s*\{\s*(\-?[0-9\.\,]+)\s*\}",       # \answer{18}
            r"\\final\s*\{\s*(\-?[0-9\.\,]+)\s*\}",        # \final{18}
            r"\\result\s*\{\s*(\-?[0-9\.\,]+)\s*\}",       # \result{18}
            r"####\s*(\-?[0-9]+)",                          # ####18 (更简单的模式)
            r"\\boxed\{(\-?[0-9]+)\}",                      # \boxed{18} (更简单的模式)
            r"\\box\{(\-?[0-9]+)\}",                        # \box{18} (更简单的模式)
            r"\\answer\{(\-?[0-9]+)\}",                     # \answer{18} (更简单的模式)
            r"\\final\{(\-?[0-9]+)\}",                      # \final{18} (更简单的模式)
            r"\\result\{(\-?[0-9]+)\}",                     # \result{18} (更简单的模式)
            r"\s*\$(\-?[0-9\.\,]+)",                        # $120,000
            # 添加更多常见的答案格式
            r"Answer:\s*(\-?[0-9\.\,]+)",                   # Answer: 123
            r"The answer is\s*(\-?[0-9\.\,]+)",            # The answer is 123
            r"Therefore,?\s*(\-?[0-9\.\,]+)",              # Therefore, 123
            r"So,?\s*(\-?[0-9\.\,]+)",                     # So, 123
            r"Final answer:\s*(\-?[0-9\.\,]+)",            # Final answer: 123
            r"Result:\s*(\-?[0-9\.\,]+)",                  # Result: 123
        ]
        for pattern in solution_variants:
            matches = re.findall(pattern, solution_str)
            if len(matches) > 0:
                final_answer = matches[-1].replace(",", "").replace("$", "").replace('\n', '').strip()
                final_answer = str(final_answer)
                # 如果清理后为空，跳过这个匹配
                if final_answer:
                    break
        

        if final_answer is None:
            print("Warning: No answer found in solution_str with standard patterns")
            # 回退机制：尝试提取文本中的最后一个数字
            numbers = re.findall(r'(\-?[0-9]+(?:\.[0-9]+)?)', solution_str)
            if numbers:
                final_answer = numbers[-1]
                print(f"Fallback: Using last number found: {final_answer}")
            else:
                print("Fallback: No numbers found in solution_str")
                return None
        

    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def compute_score(solution_str, ground_truth, method="strict", score=1):
    """The scoring function for GSM8k with reasoning behavior rewards.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
    Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    
    # 如果answer为None或空字符串，给予最低分数而不是0分
    if answer is None or answer == "":
        print("Warning: No answer found in solution_str")
        print(f"Solution string length: {len(solution_str)}")
        print(f"Solution string preview: {solution_str[:200]}...")
        # 即使没有找到答案，也给予一些推理奖励
        reasoning_bonus = _compute_reasoning_bonus(solution_str)
        keyword_bonus = _compute_keyword_bonus(solution_str)
        print(f"Reasoning bonus: {reasoning_bonus}, Keyword bonus: {keyword_bonus}")
        return 0.05 + 0.08 * reasoning_bonus + 0.02 * keyword_bonus  # 最低5%分数
    print("--------------------------------")
    print(f'answer_extract is not None: {answer}')
    print("--------------------------------")
    # Clean both answer and ground_truth for comparison
    # 先保存原始答案用于调试
    original_answer = answer
    # 清理答案中的特殊字符，保留纯数字
    answer = answer.replace(",", "").replace("$", "").replace(' ', '').replace('\n', '').replace('####', '').strip()
    # 如果清理后为空，尝试从原始答案中提取数字
    if not answer:
        import re
        numbers = re.findall(r'(\-?[0-9]+(?:\.[0-9]+)?)', original_answer)
        if numbers:
            answer = numbers[-1]
        else:
            print(f"Warning: Answer is empty after cleaning: '{original_answer}'")
            return 0.05  # 返回最低分数
    assert isinstance(answer, str), "answer is not a string"

    if ground_truth is not None:
        assert isinstance(ground_truth, str), "ground_truth is not a string"
        ground_truth_post = str(ground_truth).replace(",", "").replace("$", "").replace(' ', '').replace('\n', '').replace('####', '')
        print(f'ground_truth_post: {ground_truth_post}')
        print("--------------------------------")
        assert ground_truth_post is not None, "ground_truth_post is None"
    else:
        assert False, "ground_truth is None"

    # 尝试转换为数字进行比较
    print(f"DEBUG: About to compare answer='{answer}' with ground_truth_post='{ground_truth_post}'")
    try:
        answer_float = float(answer)
        ground_truth_float = float(ground_truth_post)
        print(f"DEBUG: Comparing numbers: {answer_float} vs {ground_truth_float}")
        # 使用数字比较
        if answer_float == ground_truth_float:
            base_score = score
            print(f"DEBUG: Numbers match! Base score = {base_score}")
        else:
            base_score = 0.0
            print(f"DEBUG: Numbers don't match! Base score = {base_score}")
    except (ValueError, TypeError) as e:
        print(f"DEBUG: Number conversion failed: {e}")
        print(f"DEBUG: Comparing strings: '{answer}' vs '{ground_truth_post}'")
        # 如果转换失败，使用字符串比较
        if answer == ground_truth_post:
            base_score = score
            print(f"DEBUG: Strings match! Base score = {base_score}")
        else:
            base_score = 0.0
            print(f"DEBUG: Strings don't match! Base score = {base_score}")
    
    print(f"DEBUG: Final base_score = {base_score}")
    
    w_base = 1/(1+0.15+0.1)
    w_reasoning = 0.15/(1+0.15+0.1)
    w_keyword = 0.1/(1+0.15+0.1)

    
    # Calculate reasoning behavior bonus
    reasoning_bonus = _compute_reasoning_bonus(solution_str)
    
    # Calculate special keyword bonus
    keyword_bonus = _compute_keyword_bonus(solution_str)
    
    # Total score = base score + reasoning bonus + keyword bonus
    total_score = w_base * base_score + w_reasoning * reasoning_bonus + w_keyword * keyword_bonus
    
    # 添加调试信息
    print(f"Score breakdown: base={base_score:.3f}, reasoning={reasoning_bonus:.3f}, keyword={keyword_bonus:.3f}")
    print(f"Weights: w_base={w_base:.3f}, w_reasoning={w_reasoning:.3f}, w_keyword={w_keyword:.3f}")
    print(f"Total score: {total_score:.3f}")
    
    # Ensure score doesn't exceed 1.0
    return min(total_score, 1.0)


def _compute_reasoning_bonus(solution_str):
    """Compute reasoning behavior bonus score"""
    reasoning_bonus = 0.0
    
    # Check for reasoning step indicators
    reasoning_indicators = [
        "first", "then", "next", "finally", "therefore", "so", "because", "since",
        "step 1", "step 2", "step 3", "steps", "calculate", "solve", "analyze",
        "given", "we can see", "it follows", "in conclusion", "thus", "hence",
        "let's", "we need to", "we have", "we get", "we obtain", "we find"
    ]
    
    found_indicators = 0
    for indicator in reasoning_indicators:
        if indicator.lower() in solution_str.lower():
            found_indicators += 1
    
    # Award bonus based on number of reasoning indicators found
    if found_indicators >= 5:
        reasoning_bonus = 0.8  
    elif found_indicators >= 3:
        reasoning_bonus = 0.5
    elif found_indicators >= 1:
        reasoning_bonus = 0.2
    
    if found_indicators >=20:
        reasoning_bonus = 0.1
    
    # Check for mathematical operations
    math_operations = ["+", "-", "*", "/", "=", "×", "÷"]
    operation_count = sum(solution_str.count(op) for op in math_operations)
    if operation_count >= 3:
        reasoning_bonus += 0.2  # Additional 0.2 points
    
    return min(reasoning_bonus, 1.0)  # Reasoning bonus capped at 1.0 points


def _compute_keyword_bonus(solution_str):
    """Compute special keyword bonus score"""
    keyword_bonus = 0.0
    
    # Special keywords and their weights
    special_keywords = {
        "because": 0.03,  # Indicates causal relationship
        "therefore": 0.03,  # Indicates conclusion
        "thus": 0.02,  # Indicates result
        "since": 0.02,  # Indicates reason
        "given": 0.02,  # Indicates basis
        "we can conclude": 0.02,  # Indicates deduction
        "it follows": 0.02,  # Indicates logical consequence
        "hence": 0.02,  # Indicates result
        "so": 0.02,  # Indicates conclusion
    }
    
    num_keywords = 0
    for keyword, weight in special_keywords.items():
        if keyword.lower() in solution_str.lower():
            keyword_bonus += weight
            num_keywords += 1

    if num_keywords >= 20:
        keyword_bonus = 1.0
    elif num_keywords >= 15:
        keyword_bonus = 0.8
    elif num_keywords >= 10:
        keyword_bonus = 0.7
    elif num_keywords >= 5:
        keyword_bonus = 0.6
    elif num_keywords >= 3:
        keyword_bonus = 0.2
    elif num_keywords >= 1:
        keyword_bonus = 0.1
    
    
    return min(keyword_bonus, 1.0)  # Keyword bonus capped at 0.1 points
