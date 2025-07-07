import json
import logging
import random
import re
import time
from typing import List, Any

import numpy as np
import sympy

from utils.openai_interface import OpenAIInterface

# Part1: 定义全局变量 constraints、init_prompt、crossover_prompt、mutation_prompt
constraints = ["+", "*", "-", "/", "max","min"]
binary_ops = ["+", "-", "*", "/"]
terminals = ["TCN", "TWT", "TNC", "TUT"]
random_num = None
#
INIT_PROMPT = """
    You are a mathematical expression generator. 
    ### Generation Rules
    1. Randomly select 2 to 4 operators from the allowed set.
    2. Shuffle the selected operators to ensure diverse orderings
    3. Construct an expression using the selected operators:
       - Use {expression_type} style.
       - Ensure variation in operand placement.
    ### Selected Operators
    {selected_ops}
    ### Allowed Variables & Constants (`terminals`)
    {terminals}
    ### Your Task
    - Generate a unique expression using randomly selected operators.
    - Ensure different structures in each response.
    - Provide no additional text in response. Format output in JSON as {{"expression": "<expression>"}}
"""
CROSSOVER_PROMPT_IDX0 = """
You are given two mathematical expressions {expression}. 

Your task is to recombine these two expressions by performing a single-point crossover, similar to the crossover operation in genetic programming.

Variable meanings:
TCT: Task Computation Time — time needed to complete the task
TWT: Task Waiting Time — time from task arrival to current time
TNC: Number of child tasks that depend on this one
TUT: Task Upload Time — time needed to transfer task data to destination

Please ensure the syntax of the expressions is valid and that the recombined expressions use only the existing terms and operators from the original expressions.

Provide no additional text in response. Format your output in JSON as:{{"expressions": ["<expression>"]}}
"""

CROSSOVER_PROMPT_IDX1 = """
You are given two mathematical expressions {expression}. 

Your task is to recombine these two expressions by performing a single-point crossover, similar to the crossover operation in genetic programming.

Variable meanings:
TCT: Task Computation Time — time needed to complete the task
TWT: Task Waiting Time — time from task arrival to current time
TNC: Number of child tasks that depend on this one
TPS: Task Propagation Speed 
ACT: Average Completion Time of Child Tasks

Please ensure the syntax of the expressions is valid and that the recombined expressions use only the existing terms and operators from the original expressions.

Provide no additional text in response. Format your output in JSON as:{{"expressions": ["<expression>"]}}
"""

MUTATION_PROMPT_IDX0 = """
The goal is to evolve the mathematical expression and create a new expression that differs in structure from the original but still follows mathematical principles.

Given the expression: {expression} 

Use the listed symbols {constraints}.

Provide no additional text in response. Format output in JSON as {{"new_expression": "<new expression>"}}
"""

MUTATION_PROMPT_IDX1 = """
The goal is to evolve the mathematical expression and create a new expression that differs in structure from the original but still follows mathematical principles.

Given the expression: {expression} 

Use the listed symbols {constraints}.

Provide no additional text in response. Format output in JSON as {{"new_expression": "<new expression>"}}
"""

# Part2: 用于检验LLM生成的表达式是否有效的相关函数
class ProtectedSqrt(sympy.Function):
    # 自定义的平方根函数
    @classmethod
    def eval(cls, x):
        """如果 x 是负数，则返回保护值 1e-6，否则返回标准的 sqrt(x)"""
        if isinstance(x, (sympy.Integer, sympy.Float)):  # 如果是数值类型
            if x < 0:
                return 1e-6  # 对于负数返回保护值
            else:
                return sympy.sqrt(x)  # 对于非负数，返回标准的平方根
        return None  # 对于符号表达式，返回 None 以便进行符号化计算

    @staticmethod
    def _latex(self, printer, *args):
        """定义自定义平方根的 LaTeX 输出格式"""
        return r"\text{protected\_sqrt}(" + printer.doprint(self.args[0]) + r")"

def convert_square_to_root(expression):
    result = ""
    i = 0
    while i < len(expression):
        if expression[i:i + 7] == "square(":
            start = i + 7
            stack = 1
            j = start
            while j < len(expression) and stack > 0:
                if expression[j] == '(':
                    stack += 1
                elif expression[j] == ')':
                    stack -= 1
                j += 1
            inner_expression = expression[start:j - 1]
            result += f"root({inner_expression}, 1/2)"
            i = j
        else:
            result += expression[i]
            i += 1
    return result

def is_valid_expression(expression: str, idx: int) -> bool:
    try:
        # 1. 替换 square(x) -> root(x, 1/2)
        expression = convert_square_to_root(expression)

        # 2. 替换 sqrt(x) -> ProtectedSqrt(x)
        expression = re.sub(r'sqrt\((.*?)\)', r'ProtectedSqrt(\1)', expression)

        # 3. 定义变量集
        if idx == 0:
            TCT, TWT, TNC, TUT = sympy.symbols('TCT TWT TNC TUT')
            symbol_map = {TCT: 1.1, TWT: 1.2, TNC: 1.3, TUT: 1.4}
        elif idx == 1:
            TCT, TWT, TNC, TPS, ACT = sympy.symbols('TCT TWT TNC TPS ACT')
            symbol_map = {TCT: 1.1, TWT: 1.2, TNC: 1.3, TPS: 1.4, ACT: 1.5}
        else:
            raise ValueError(f"Unsupported idx: {idx}")

        # 4. 解析表达式
        expr = sympy.sympify(expression, evaluate=False, locals={'ProtectedSqrt': ProtectedSqrt})

        # 5. 代入数值
        result = expr.subs(symbol_map)

        # 6. 检查是否为实数
        return result.is_real

    except sympy.SympifyError as e:
        logging.error(f"SympifyError: {e} for expression {expression}")
    except TypeError as e:
        logging.error(f"TypeError: {e} for expression {expression}")
    except Exception as e:
        logging.error(f"Unexpected Error: {e} for expression {expression}")

    return False


def check_response_individual_generation(response: str) -> str:
    # 解析LLM生成的JSON响应，提取表达式
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if not match:
        return "0"
    json_text = match.group(0)
    try:
        expression = json.loads(json_text).get('expression', None)
        if expression is not None and is_valid_expression(expression):
            return expression
        else:
            return "0"
    except json.decoder.JSONDecodeError:
        return "0"


# def check_response_crossover(response: str, parents: List[str]) -> List[str]:
#     """
#     解析 LLM 生成的 `response`，提取 `expressions`，如果解析失败，则返回 `parents` 作为备选值。
#     """
#     try:
#         response_cleaned = re.sub(r'```json\n|```', '', response).replace('\n', '')
#
#         match = re.search(r'\"expressions\"\s*:\s*\[(.*?)\]', response_cleaned, re.DOTALL)
#         if not match:
#             return parents
#
#         expressions_string = match.group(1)
#
#         expressions = re.findall(r'\{([^\}]+)\}', expressions_string)  # 提取 `{}` 内的表达式
#         if expressions:
#             cleaned_exprs = [expr.strip().replace('"', '') for expr in expressions]
#             return cleaned_exprs if len(cleaned_exprs) == 2 else parents
#
#         final_exprs = [expr.strip().replace('"', '').replace("{", "").replace("}", "")
#                        for expr in re.split(r'\s*,\s*', expressions_string)]
#
#         return final_exprs if len(final_exprs) == 2 else parents  # **确保返回两个表达式**
#
#     except (ValueError, json.JSONDecodeError) as e:
#         logging.error(f"解析 LLM 交叉变异响应失败: {e}")
#         return parents  # **解析失败时，返回 `parents` 作为默认值**

def check_response_crossover(response: str, parents: List[str]) -> List[str]:
    """
    健壮地从 LLM 的非标准 JSON 响应中提取 expressions（字符串列表），
    避免使用 split(',') 导致的表达式拆分错误。
    """
    try:
        # Step 1: 清洗 markdown 包装
        response_cleaned = re.sub(r'```json\n|```', '', response).strip()

        # Step 2: 提取 expressions 字符串部分
        match = re.search(r'"expressions"\s*:\s*\[(.*?)\]', response_cleaned, re.DOTALL)
        if not match:
            print("❌ 未能匹配 expressions 字段")
            return parents

        expressions_block = match.group(1)

        # Step 3: 用正则提取每个被引号包裹的完整表达式
        expr_matches = re.findall(r'"(.*?)"', expressions_block)
        for idx, expr in enumerate(expr_matches):
            print(f"  Expression {idx+1}: {expr}")

        if len(expr_matches) == 2:
            print("✅ 成功提取 2 个表达式！")
            return expr_matches
        else:
            print(f"⚠️ 提取到 {len(expr_matches)} 个表达式，使用 parents 作为备选。")
            return parents

    except Exception as e:
        logging.error(f"解析 LLM 交叉表达式失败: {e}")
        print(f"🚨 解析出错: {e}")
        return parents


def check_mutation_response(response: str, expression: str) -> str:
    """
    健壮地从 LLM 的非标准 JSON 响应中提取 new_expression 字段，并验证其合法性。
    """
    try:
        print("\n📥 原始 response：")
        print(response)

        # Step 1: 清洗 markdown 包装符
        cleaned_content = re.sub(r'```json\n|```', '', response).strip()
        cleaned_content = cleaned_content.replace('\n', '')
        print("\n📃 清洗后的 cleaned_content：")
        print(cleaned_content)

        # Step 2: 使用正则提取 new_expression 字段
        match = re.search(r'"new_expression"\s*:\s*"(.*?)"', cleaned_content, re.DOTALL)
        new_expression = match.group(1).strip().replace('"', '') if match else None

        # Step 3: 打印提取结果
        if new_expression:
            print(f"\n🔍 提取出的 new_expression：{new_expression}")
        else:
            print("❌ 未能提取到 new_expression 字段")

    except (ValueError, json.JSONDecodeError) as e:
        logging.error(f"解析 LLM 变异响应时出错: {e}")
        print(f"🚨 解析异常: {e}")
        return expression


# Part3: 生成init、 crossover、mutation的提示词
def form_prompt_generation(init_prompt) -> str:
    global random_num  # 使用全局变量来追踪上一次的随机数

    # 生成新的 0-1 之间的随机数，并保留两位小数
    new_random_num = str(round(np.random.uniform(0, 1), 2))

    # 如果之前已经有一个随机数，先移除它
    if random_num is not None:
        terminals.remove(random_num)

    # 添加新生成的随机数
    terminals.append(new_random_num)

    # 更新全局变量
    random_num = new_random_num

    num_ops = random.randint(2, 4)
    selected_ops = random.sample(constraints, num_ops)

    # **Ensure at least one binary operator**
    binary_ops = ["+", "-", "*", "/"]
    if not any(op in selected_ops for op in binary_ops):
        selected_ops.append(random.choice(binary_ops))

    # **50% chance to generate `genFull()`, 50% chance to generate `genGrow()`**
    expression_type = "fully-expanded tree (genFull)" if random.random() < 0.5 else "random-growth tree (genGrow)"

    prompt = init_prompt.format(
        expression_type=expression_type,
        selected_ops=selected_ops,
        terminals=terminals,
    )
    # print(f"---------------initial prompt: {prompt}-----------------")

    return prompt

def form_llm_crossover_expressions(expressions, crossover_prompt,) -> str:
    expressions = " and ".join(expressions)
    prompt = crossover_prompt.format(
        expression=expressions,
    )
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print(f"crossover prompt:{prompt}")
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    return prompt

def form_prompt_rephrase_mutation(
        expression: str, mutation_prompt,
) -> str:

    prompt = mutation_prompt.format(
        expression=expression,
        constraints=constraints,
    )
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print(f"mutation prompt:{prompt}")
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    return prompt

# Part4: 调用LLM，发送请求、接收响应
def collect_llm_generate_expressions(llm_interface: OpenAIInterface, generation_history: list, population_size: int) -> list:
    # 收集LLM生成的表达式，只包括有效的表达式
    expressions = []
    for i in range(population_size):
        # history_prompt = "\n".join(
        #     [f"Generated Expression {idx + 1}: {expr['expression']}" for idx, expr in enumerate(generation_history)]
        # )
        prompt = form_prompt_generation(INIT_PROMPT)
        response = llm_interface.predict_text_logged(prompt, temp=1)
        #
        expression = check_response_individual_generation(response["content"])

        # **存入 `generation_history`**
        generation_history.append({"expression": expression})
        expressions.append(expression)

        print(f"LLM 生成的最终表达式: {expression}")

    return expressions

def llm_crossover_expressions(
    llm_interface: OpenAIInterface,
    parents: List[str],
    idx: int
) -> List[str]:
    children = parents[:]  # 默认子代继承父代

    # Step 1: 选择对应提示词
    if idx == 0:
        prompt = form_llm_crossover_expressions(parents, CROSSOVER_PROMPT_IDX0)
    elif idx == 1:
        prompt = form_llm_crossover_expressions(parents, CROSSOVER_PROMPT_IDX1)
    else:
        raise ValueError(f"不支持的 idx 值: {idx}")

    print("📝 交叉操作的提示词为:\n", prompt)

    # Step 2: 发送至 LLM 获取响应
    response = llm_interface.predict_text_logged(prompt, temp=1.0)

    print("\n📨 LLM 返回原始 response[\"content\"]：")
    print(response["content"])

    # Step 3: 尝试解析 expressions
    new_expressions = check_response_crossover(response["content"], parents)

    # Step 4: 验证表达式合法性
    if len(new_expressions) == 2:
        valid_flags = [is_valid_expression(expr, idx) for expr in new_expressions]

        for i, (expr, is_valid) in enumerate(zip(new_expressions, valid_flags)):
            print(f"✅ 子表达式 {i+1} 是否合法: {is_valid} —— 内容: {expr}")

        if all(valid_flags):
            print("🎉 两个表达式都合法，接受为子代")
            children = new_expressions
        else:
            print("⚠️ 表达式语义验证失败，保持原父代表达式")
    else:
        print(f"❌ 表达式数量为 {len(new_expressions)}，应为 2，保持原父代表达式")

    print(f"\n👶 最终返回的子代表达式: {children}")
    return children

def llm_mutated_expressions(
        llm_interface: OpenAIInterface,
        expression: str,
        idx: int
) -> str:

    # Step 1: 构建提示词（根据 idx 选择不同模板）
    if idx == 0:
        prompt = form_prompt_rephrase_mutation(expression, MUTATION_PROMPT_IDX0)
    elif idx == 1:
        prompt = form_prompt_rephrase_mutation(expression, MUTATION_PROMPT_IDX1)
    else:
        raise ValueError(f"不支持的 idx 值: {idx}")

    print("📝 变异操作的提示词为:\n", prompt)

    # Step 2: 调用 LLM
    response = llm_interface.predict_text_logged(prompt, temp=1)

    print("\n📨 LLM 返回原始 response['content']：")
    print(response["content"])

    # Step 3: 提取表达式
    new_expression = check_mutation_response(response["content"],expression)

    # Step 4: 表达式合法性验证
    if new_expression and is_valid_expression(new_expression, idx):
        print(f"✅ 新表达式合法，作为变异结果：{new_expression}")
        return new_expression
    else:
        print("⚠️ 表达式无效或提取失败，使用原表达式")
        return expression
