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

"""
Enhanced reward function for character counting task
"""

import re


def extract_boxed_answer(text):
    """
    Extract answer from \\boxed{} format
    """
    # 匹配 \boxed{...} 或 \boxed{...} 格式
    boxed_patterns = [
        r'\\boxed\{([^}]+)\}',
        r'\\boxed\{([^}]+)\}',
        r'boxed\{([^}]+)\}',
        r'boxed\{([^}]+)\}'
    ]
    
    for pattern in boxed_patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()  # 返回最后一个匹配
    return None


def extract_numbers(text):
    """
    Extract all numbers from text
    """
    numbers = re.findall(r'\b\d+\b', text)
    return [int(num) for num in numbers]


def char_count_reward_function(data_source, solution_str, ground_truth, extra_info=None):
    """
    Enhanced reward function for character counting task.
    
    Args:
        data_source: Source of the data
        solution_str: Model's response string
        ground_truth: Expected correct answer
        extra_info: Additional information (optional)
    
    Returns:
        float: Reward score between 0 and 1
    """
    try:
        # 清理输入
        solution_str = str(solution_str).strip()
        ground_truth = str(ground_truth).strip()
        
        if not solution_str:
            return 0.0
        
        # 1. 首先尝试从boxed格式中提取答案
        boxed_answer = extract_boxed_answer(solution_str)
        if boxed_answer:
            if boxed_answer == ground_truth:
                return 1.0  # 完全正确
            else:
                # 检查是否为数字格式
                try:
                    solution_num = int(boxed_answer)
                    ground_truth_num = int(ground_truth)
                    if solution_num == ground_truth_num:
                        return 1.0
                except ValueError:
                    pass
        
        # 2. 如果没有boxed格式，尝试从文本中提取数字
        numbers = extract_numbers(solution_str)
        if numbers:
            last_number = numbers[-1]
            try:
                ground_truth_num = int(ground_truth)
                if last_number == ground_truth_num:
                    return 0.8  # 数字正确但没有boxed格式
            except ValueError:
                pass
        
        # 3. 检查是否包含正确的推理步骤
        reasoning_score = 0.0
        
        # 检查是否包含字符比较步骤
        if "=" in solution_str or "!=" in solution_str:
            reasoning_score += 0.2
        
        # 检查是否包含目标字符
        if extra_info and "response" in extra_info:
            # 从原始响应中提取目标字符
            original_response = extra_info["response"]
            if "How many" in original_response:
                # 尝试提取目标字符
                char_match = re.search(r'How many (\w) are there in', original_response)
                if char_match:
                    target_char = char_match.group(1)
                    if target_char in solution_str:
                        reasoning_score += 0.1
        
        # 4. 检查答案格式
        format_score = 0.0
        if "\\boxed{" in solution_str or "\\boxed{" in solution_str:
            format_score += 0.1
        
        # 5. 检查是否有合理的答案范围
        try:
            ground_truth_num = int(ground_truth)
            if 0 <= ground_truth_num <= 20:  # 合理的字符计数范围
                numbers_in_response = extract_numbers(solution_str)
                for num in numbers_in_response:
                    if 0 <= num <= 20:
                        format_score += 0.05
                        break
        except ValueError:
            pass
        
        # 6. 检查是否包含计数相关的关键词
        count_keywords = ['count', 'occurrence', 'times', 'number', 'total']
        if any(keyword in solution_str.lower() for keyword in count_keywords):
            reasoning_score += 0.05
        
        # 7. 计算总奖励
        total_reward = reasoning_score + format_score
        
        # 确保奖励在合理范围内
        return min(total_reward, 1.0)
        
    except Exception as e:
        # 记录错误但不中断训练
        print(f"Error in reward function: {e}")
        print(f"Ground truth: {ground_truth}")
        print(f"Solution: {solution_str}")
        return 0.0


def char_count_reward_function_simple(data_source, solution_str, ground_truth, extra_info=None):
    """
    Simplified version of the reward function for basic testing.
    """
    try:
        # 清理输入
        solution_str = str(solution_str).strip()
        ground_truth = str(ground_truth).strip()
        
        if not solution_str:
            return 0.0
        
        # 提取boxed答案
        boxed_answer = extract_boxed_answer(solution_str)
        if boxed_answer:
            # 直接比较答案
            if boxed_answer == ground_truth:
                return 1.0
            else:
                # 尝试数字比较
                try:
                    solution_num = int(boxed_answer)
                    ground_truth_num = int(ground_truth)
                    if solution_num == ground_truth_num:
                        return 1.0
                except ValueError:
                    pass
        
        # 如果没有boxed格式，尝试从文本中提取数字
        numbers = extract_numbers(solution_str)
        if numbers:
            last_number = numbers[-1]
            try:
                ground_truth_num = int(ground_truth)
                if last_number == ground_truth_num:
                    return 0.5  # 数字正确但没有boxed格式
            except ValueError:
                pass
            
        return 0.1  # 小奖励鼓励尝试
        
    except Exception:
        return 0.0


def char_count_reward_function_strict(data_source, solution_str, ground_truth, extra_info=None):
    """
    Strict version that only rewards perfect answers with boxed format.
    """
    try:
        # 清理输入
        solution_str = str(solution_str).strip()
        ground_truth = str(ground_truth).strip()
        
        if not solution_str:
            return 0.0
        
        # 只检查boxed格式
        boxed_answer = extract_boxed_answer(solution_str)
        if not boxed_answer:
            return 0.0
        
        # 严格比较
        if boxed_answer == ground_truth:
            return 1.0
        
        # 尝试数字比较
        try:
            solution_num = int(boxed_answer)
            ground_truth_num = int(ground_truth)
            if solution_num == ground_truth_num:
                return 1.0
        except ValueError:
            pass
        
        return 0.0
        
    except Exception:
        return 0.0
