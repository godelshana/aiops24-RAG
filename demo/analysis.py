import re
from typing import List, Dict, Any

def extract_evaluation_info(file_path: str) -> List[Dict[str, Dict[str, Any]]]:
    """
    从文件中提取评估信息，使用更灵活的正则表达式并添加调试信息
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()


    # 尝试匹配整个评估结果结构
    pattern = r"'(\w+)':\s*EvaluationResult\((.*?),\s*(?:invalid_result=\w+,\s*invalid_reason=\w+)?\)"
    matches = re.findall(pattern, content, re.DOTALL)

    results = []
    num = 0
    for eval_type, eval_content in matches:
        print(f"\n找到评估类型: {eval_type}")
        info = {}
        
        # 提取各个字段
        score_match = re.search(r"score=([\d.]+)", eval_content)
        passing_match = re.search(r"passing=(True|False)", eval_content)
        feedback_match = re.search(r"feedback='([^']*)'", eval_content)
        # print(eval_content, score_match, passing_match, feedback_match)

        if score_match:
            info['score'] = float(score_match.group(1))
            info['id'] = num
            num += 1
            print(f"  分数: {info['score']}")
        if passing_match:
            info['passing'] = passing_match.group(1) == 'True'
            print(f"  通过: {info['passing']}")
        if feedback_match:
            info['feedback'] = feedback_match.group(1)
            print(f"  反馈: {info['feedback'][:50]}...")  # 只打印前50个字符
        results.append({eval_type: info})

    print(f"\n总共找到 {len(results)} 个评估结果")
    return results

def analyze_evaluation_results(results: List[Dict[str, Dict[str, Any]]]) -> str:
    """
    分析评估结果并生成总结报告
    """
    analysis = "评估结果分析：\n\n"
    for result in results:
        for eval_type, info in result.items():
            analysis += f"{eval_type.capitalize()} 评估:\n"
            analysis += f"  分数: {info.get('score', 'N/A')}\n"
            analysis += f"  通过: {'是' if info.get('passing') else '否'}\n"
            analysis += f"  反馈: {info.get('feedback', 'N/A')}\n"
            analysis += "\n"
    
    valid_scores = [info['score'] for result in results for info in result.values() if 'score' in info]
    if valid_scores:
        overall_score = sum(valid_scores) / len(valid_scores)
        analysis += f"总体评分: {overall_score:.2f}\n"
    else:
        analysis += "无法计算总体评分（没有有效分数）\n"
    
    return analysis

def summarize_results(results):
    faithfulness_total = 0
    faithfulness_pass = 0
    relevancy_total = 0
    relevancy_pass = 0
    correctness_scores = []
    
    for result in results:
        for eval_type, info in result.items():
            if eval_type == 'faithfulness':
                faithfulness_total += 1
                if info.get('passing', False):
                    faithfulness_pass += 1
            elif eval_type == 'relevancy':
                relevancy_total += 1
                if info.get('passing', False):
                    relevancy_pass += 1
            elif eval_type == 'correctness':
                correctness_scores.append((info['id'], info.get('score', 0)))
    
    # 计算Correctness的平均分
    avg_correctness = sum(score for _, score in correctness_scores) / len(correctness_scores) if correctness_scores else 0
    
    # 找出Correctness倒数前十的分数及其id
    correctness_scores.sort(key=lambda x: x[1])  # 按分数升序排序
    bottom_ten = correctness_scores[:10] if len(correctness_scores) > 10 else correctness_scores
    
    print(f"Faithfulness总数: {faithfulness_total}")
    print(f"Faithfulness通过数量: {faithfulness_pass}")
    print(f"Relevancy总数: {relevancy_total}")
    print(f"Relevancy通过数量: {relevancy_pass}")
    print(f"Correctness平均分: {avg_correctness:.2f}")
    print("Correctness倒数前十分数的id:")
    for id, score in bottom_ten:
        print(f"  ID: {id}, 分数: {score}")

def main(file_path: str):
    # 提取评估信息
    results = extract_evaluation_info(file_path)
    
    # 分析结果
    analysis = analyze_evaluation_results(results)

    print(summarize_results(results))
    
    # 打印分析结果
    # print(analysis)
    
    # 可选：将分析结果保存到文件
    with open('evaluation_analysis1.txt', 'w', encoding='utf-8') as file:
        file.write(analysis)

if __name__ == "__main__":
    file_path = "eval1.txt"  # 请替换为实际的文件路径
    main(file_path)