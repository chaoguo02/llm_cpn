import ast

from deap import creator, gp


def expression_to_tree(expr):
    tree = ast.parse(expr, mode='eval')

    def convert(node):
        if isinstance(node, ast.BinOp):
            op_name = {
                ast.Add: "add",
                ast.Sub: "sub",
                ast.Mult: "mul",
                ast.Div: "protect_div",
                ast.Pow: "pow"
            }[type(node.op)]
            return f"{op_name}({convert(node.left)}, {convert(node.right)})"
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return f"neg({convert(node.operand)})"
        elif isinstance(node, ast.Call):
            func_name = {
                "sqrt": "protect_sqrt",
                "sin": "sin",
                "cos": "cos",
                "exp": "exp",
                "log": "log",
                "square": "square",
                "min": "min",
                "max": "max",
            }.get(node.func.id, node.func.id)
            args = ", ".join(convert(arg) for arg in node.args)
            return f"{func_name}({args})"
        elif isinstance(node, ast.Num):
            return str(node.n)
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.BoolOp):
            raise ValueError(f"❌ 不支持布尔运算符 (and/or)，表达式中出现了逻辑操作: {ast.dump(node)}")
        else:
            raise ValueError(f"Unsupported AST node: {ast.dump(node)}")

    return convert(tree.body)


def tree_to_expression(tree):
    """
    优化 GP 生成的 `PrimitiveTree` 结构，转换为更 **简化** 的数学表达式。

    主要优化：
    1. **去除不必要的括号**
    2. **根据运算符优先级简化表达式**
    """

    if isinstance(tree, creator.Individual):
        tree = gp.PrimitiveTree(tree)

    if isinstance(tree, gp.PrimitiveTree):
        nodes = list(tree)  # 线性化 PrimitiveTree
    else:
        raise ValueError(f"Unsupported node type: {type(tree)}")

    # **运算符优先级**（数值越高，优先级越高）
    precedence = {
        "+": 2,
        "-": 2,
        "*": 3,
        "/": 3,
        "**": 4,  # 指数运算符最高
        "sqrt": 5, "square": 5, "sin": 5, "cos": 5, "exp": 5, "log": 5, "neg": 5,  # 一元运算符最高
        "min": 1,
        "max": 1
    }

    stack = []

    for node in reversed(nodes):  # 逆序遍历 GP 树
        if isinstance(node, gp.Primitive):  # 处理运算符
            func_map = {
                "add": "+",
                "sub": "-",
                "mul": "*",
                "protect_div": "/",
                "pow": "**",
                "neg": "-",
                "protect_sqrt": "sqrt",
                "square": "square",
                "sin": "sin",
                "cos": "cos",
                "exp": "exp",
                "log": "log",
                "min": "min",
                "max": "max"
            }
            func_name = node.name
            op = func_map.get(func_name, func_name)

            if node.arity == 1:  # **一元运算符**
                operand = stack.pop()
                if precedence[op] < precedence.get(operand, 100):
                    stack.append(f"{op}({operand})")  # 括号必须保留
                else:
                    stack.append(f"{op} {operand}")  # 省略括号

            elif node.arity == 2:  # **二元运算符**
                left = stack.pop()
                right = stack.pop()

                # **判断是否需要括号**
                if op in {"min", "max"}:
                    stack.append(f"{op}({left}, {right})")
                else:
                    left_expr = f"({left})" if precedence[op] > precedence.get(left[0], 100) else left
                    right_expr = f"({right})" if precedence[op] > precedence.get(right[0], 100) else right

                    stack.append(f"{left_expr} {op} {right_expr}")

            else:
                raise ValueError(f"Unsupported function arity: {func_name}({node.arity})")

        elif isinstance(node, gp.Terminal):  # **变量或常数**
            stack.append(str(node.value))

        else:
            raise ValueError(f"Unsupported node type: {type(node)}")

    return stack[0]  # 返回最终简化表达式
