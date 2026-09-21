import ast
import inspect
import re
from ast import expr

import ast_comments
from black import Mode, format_str
from sympy import Add, Mul, Pow, Eq, Not, Xor

from expr_codegen.expr import register_symbols, list_to_exprs


class ParentSetter(ast.NodeVisitor):
    """为 AST 节点添加父节点引用
    
    高级用法：
    1. 继承 ast.NodeVisitor 实现 AST 遍历
    2. ast.iter_child_nodes 迭代子节点
    3. 动态添加 _parent 属性，建立父子关系
    
    应用场景：
    - 在后续转换中需要知道节点的父节点类型
    - 例如判断一个 Name 节点是否是函数调用的函数名
    
    注意：
    - Python 的 AST 默认不存储父节点引用
    - 需要手动遍历添加，这是 AST 处理的常见模式
    """

    def visit(self, node):
        # 为每个子节点添加 _parent 属性，指向当前节点
        for child in ast.iter_child_nodes(node):
            child._parent = node
        return super().visit(node)


class KwargsTransformer(ast.NodeTransformer):
    """将代码中的命名参数替换为常量
    
    高级用法：
    1. 继承 ast.NodeTransformer 实现 AST 转换
    2. 检查节点上下文（ast.Load）避免替换赋值目标
    3. 使用 _parent 属性判断节点是否为函数名
    
    应用场景：
    - 实现模板代码的参数化
    - 例如将函数中的 'date' 参数替换为具体值
    
    Parameters
    ----------
    kwargs : dict
        参数名到值的映射
    """

    def __init__(self, kwargs):
        self.kwargs = kwargs

    def visit_Name(self, node):
        # 如果变量名不在替换列表中，跳过
        if node.id not in self.kwargs:
            return node
        # 如果不是读取上下文（如赋值左侧），跳过
        if not isinstance(node.ctx, ast.Load):
            return node

        # 获取父节点
        parent = getattr(node, '_parent', None)
        # 如果是函数调用的函数名，不替换
        if isinstance(parent, ast.Call) and parent.func is node:
            return node

        # 替换为常量
        return ast.Constant(value=self.kwargs.get(node.id))


def code_format(code, **kwargs) -> str:
    """将代码块中的部分命名参数替换成常量，实现动态参数
    
    高级用法：
    1. inspect.getsource 获取函数源代码
    2. ast_comments.parse 解析带注释的 AST
    3. 使用 Transformer 链进行多次转换
    4. ast_comments.unparse 将 AST 转回代码字符串
    
    应用场景：
    - 模板代码生成，将占位符替换为实际值
    - 例如：将 'date' 参数替换为 '2023-01-01'
    
    Parameters
    ----------
    code : callable or str
        函数对象或代码字符串
    **kwargs : dict
        要替换的参数映射
    
    Returns
    -------
    str
        替换后的代码字符串
    """
    # 获取代码源并解析为 AST
    tree = ast_comments.parse(inspect.getsource(code))
    # 第一步：为所有节点添加父节点引用
    t1 = ParentSetter()
    t1.visit(tree)
    # 第二步：替换参数为常量
    t2 = KwargsTransformer(kwargs)
    t2.visit(tree)
    # 将 AST 转回代码字符串
    return ast_comments.unparse(tree)


class SyntaxTransformer(ast.NodeTransformer):
    """修改 Python 语法为自定义 DSL 语法
    
    高级用法：
    1. 继承 ast.NodeTransformer 实现 AST 语法转换
    2. 支持多种语法糖：三元表达式、位运算、下标等
    3. 转换顺序很重要：先语法转换，再变量重命名
    
    转换规则：
    - 三元表达式 a if b else c -> if_else(b, a, c)
    - 位运算 ^ -> Pow 或 Xor
    - 下标 x[1] -> ts_delay(x, 1)
    - 按位取反 ~x -> Not(x)
    
    Parameters
    ----------
    convert_xor : bool
        True: ^ 转为乘方 Pow
        False: ^ 转为异或 Xor
    """

    def __init__(self, convert_xor):
        # ^ 是异或还是乘方呢？
        self.convert_xor = convert_xor

    def visit_Assign(self, node):
        """处理赋值语句，将元组解包转换为 unpack 调用
        
        转换示例：
        输入: A, B, C = MACD()
        输出:
            A = unpack(MACD(), 0)
            B = unpack(MACD(), 1)
            C = unpack(MACD(), 2)
        
        高级用法：
        1. 检测赋值目标是否为元组 (ast.Tuple)
        2. 为每个元组元素生成一个独立的赋值语句
        3. 使用 unpack 函数提取元组的对应索引
        
        为什么需要转换：
        - DAG 无法直接处理元组解包
        - 转换后 CSE 可以优化公共子表达式
        - 配合 GlobalVariable 处理 DataFrame 无法存储 tuple 的问题
        
        Parameters
        ----------
        node : ast.Assign
            赋值语句节点
        
        Returns
        -------
        list or ast.Assign
            如果是元组解包，返回多个赋值语句列表
            否则返回原节点
        """
        t = node.targets[0]  # 获取赋值目标（左值）
        nodes = []
        if isinstance(t, ast.Tuple):  # 检测是否为元组解包
            # 遍历元组的每个元素
            for i, dim in enumerate(t.dims):
                # 生成 unpack 函数调用
                # 相当于: unpack(原值, 索引)
                _v = ast.Call(
                    func=ast.Name(id='unpack', ctx=ast.Load()),  # 函数名: unpack
                    args=[node.value, ast.Constant(i)],  # 参数: 原值和索引
                    keywords=[],
                )
                # 生成新的赋值语句
                # 相当于: dim = unpack(原值, 索引)
                n = ast.Assign([dim], _v, ctx=ast.Load())
                nodes.append(n)
            return nodes  # 返回多个赋值语句

        self.generic_visit(node)  # 递归处理子节点
        return node

    def visit_IfExp(self, node):
        """转换三元表达式为函数调用
        
        高级用法：
        1. 检测特殊的三元表达式模式（or True）
        2. 交换 test 和 body 实现条件反转
        3. 转换为 if_else 函数调用
        
        转换流程：
        场景1：正常三元表达式
        - 用户输入: a if b else c
        - source_replace: 不处理
        - AST: IfExp(test=b, body=a, orelse=c)
        - 转换: if_else(b, a, c)
        
        场景2：自定义语法 C?T:F
        - 用户输入: C?T:F
        - source_replace: C or True if( T )else F
        - AST: IfExp(test=T, body=C or True, orelse=F)
        - 详细 AST 结构:
          Module(
            body=[
              Expr(
                value=IfExp(
                  test=Name(id='T', ctx=Load()),
                  body=BoolOp(
                    op=Or(),
                    values=[
                      Name(id='C', ctx=Load()),
                      Constant(value=True)]),
                  orelse=Name(id='F', ctx=Load())))],
            type_ignores=[])
        - 检测到 or True 标记，交换: IfExp(test=C, body=T, orelse=F)
        - 转换: if_else(C, T, F)
        
        嵌套场景：A?B?C:D:E
        - source_replace: A or True if( B or True if( C )else D )else E
        - AST: 外层 IfExp 的 orelse 部分包含内层 IfExp
        - visit 顺序: 前序遍历，先访问外层，再访问内层
        - 转换顺序: 先转换内层，再转换外层（因为 generic_visit 会递归）
        
        注意：
        - or True 是 source_replace 添加的特殊标记
        - 标记表示这个三元表达式来自 C?T:F 语法
        - 需要交换 test 和 body 恢复原始语义
        """
        # 三元表达式。需要在外部提前替换成or True if else
        # 只要body区域，出现了or True，就认为是特殊处理过的
        # node.body 是三元表达式的真值部分（条件为真时返回的值）
        # 在场景2中，C?T:F 转换为 C or True if( T )else F
        # 此时 node.body = C or True，这是一个 BoolOp 节点
        if isinstance(node.body, ast.BoolOp) and isinstance(node.body.op, ast.Or):
            # BoolOp 是布尔运算符节点，表示 or/and 等布尔运算
            # node.body.op 是具体的运算符，这里是 ast.Or（或运算）
            # node.body.values 是 BoolOp 的操作数列表
            # 对于 "C or True"，values = [C, True]
            # node.body.values[0] 是第一个操作数 C
            # node.body.values[-1] 是最后一个操作数 True
            if isinstance(node.body.values[-1], ast.Constant):
                # 检查最后一个操作数是否为常量（True/False/数字等）
                if node.body.values[-1].value:
                    # 检查常量的值是否为真值（True、非零数字等）
                    # 如果是，说明这是 source_replace 添加的特殊标记
                    # 交换 test 和 body，实现条件反转
                    # 例如: C or True if( T )else F -> T if C else F
                    # node.test 原来是 T，现在变成 C（node.body.values[0]）
                    # node.body 原来是 C or True，现在变成 T（原来的 test）
                    node.test, node.body = node.body.values[0], node.test

        # 转换为 if_else 函数调用
        # 参数顺序: 条件, 真值, 假值
        node = ast.Call(
            func=ast.Name(id='if_else', ctx=ast.Load()),
            args=[node.test, node.body, node.orelse],
            keywords=[],
        )

        self.generic_visit(node)
        return node

    def visit_BinOp(self, node):
        """转换二元运算符
        
        高级用法：
        1. 处理比较表达式参与算术运算的情况
        2. 将位运算符转换为函数调用
        3. 使用 int_ 包装比较表达式
        
        转换示例：
        - (OPEN < CLOSE) * -1 -> int_(OPEN < CLOSE) * -1
        - a ^ b -> Pow(a, b) 或 Xor(a, b)
        
        注意：
        - 比较表达式不能直接参与算术运算，需要 int_ 转换
        - ^ 在 Python 中是异或，在数学表达式中常表示乘方
        """
        # TypeError: unsupported operand type(s) for *: 'StrictLessThan' and 'int'
        if isinstance(node.op, (ast.Mult, ast.Add, ast.Div, ast.Sub)):
            # (OPEN < CLOSE) * -1
            if isinstance(node.left, ast.Compare):
                node.left = ast.Call(
                    func=ast.Name(id='int_', ctx=ast.Load()),
                    args=[node.left],
                    keywords=[],
                )
            # -1*(OPEN < CLOSE)
            if isinstance(node.right, ast.Compare):
                node.right = ast.Call(
                    func=ast.Name(id='int_', ctx=ast.Load()),
                    args=[node.right],
                    keywords=[],
                )
            # 这种情况，已经包含
            # (OPEN < CLOSE)*(OPEN < CLOSE)

        if isinstance(node.op, ast.BitXor):
            # ^ 运算符，转换为pow还是xor
            if self.convert_xor:
                node = ast.Call(
                    func=ast.Name(id='Pow', ctx=ast.Load()),
                    args=[node.left, node.right],
                    keywords=[],
                )
            else:
                node = ast.Call(
                    func=ast.Name(id='Xor', ctx=ast.Load()),
                    args=[node.left, node.right],
                    keywords=[],
                )

        self.generic_visit(node)
        return node

    def visit_UnaryOp(self, node):
        # ~ts_delay 报错，替换成Not(ts_delay)
        if isinstance(node.op, ast.Invert):
            node = ast.Call(
                func=ast.Name(id='Not', ctx=ast.Load()),
                args=[node.operand],
                keywords=[],
            )

        self.generic_visit(node)
        return node

    def visit_Subscript(self, node):
        """转换下标操作为时序延迟函数
        
        高级用法：
        1. x[0] 直接简化为 x（当前值）
        2. x[-0] 也简化为 x
        3. x[n] 转换为 ts_delay(x, n)
        
        转换示例：
        - CLOSE[0] -> CLOSE
        - CLOSE[1] -> ts_delay(CLOSE, 1)
        - CLOSE[-1] -> ts_delay(CLOSE, -1)
        
        注意：
        - 这是 WorldQuant Alpha101 风格的时序延迟语法
        """
        if isinstance(node.slice, ast.Constant) and node.slice.value == 0:
            # x[0] 直接返回 x
            node = node.value
        elif isinstance(node.slice, ast.UnaryOp) and isinstance(node.slice.operand,
                                                                ast.Constant) and node.slice.operand.value == 0:
            # x[-0] 也直接返回 x
            node = node.value
        else:
            # x[n] 转换为 ts_delay(x, n)
            node = ast.Call(
                func=ast.Name(id='ts_delay', ctx=ast.Load()),
                args=[node.value, node.slice],
                keywords=[],
            )
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node):
        return node

    def visit_ClassDef(self, node):
        return node

    def visit_Import(self, node):
        return node

    def visit_ImportFrom(self, node):
        return node


class RenameTransformer(ast.NodeTransformer):
    """变量和函数重命名
    
    高级用法：
    1. 支持函数名、参数名、目标变量名的映射
    2. 自动处理 Python 保留字（True/False/None）
    3. 下划线开头的变量自动重命名避免冲突
    
    重命名规则：
    - True -> _TRUE_
    - False -> _FALSE_
    - None -> _NONE_
    - _变量 -> _变量_数字_ （避免与 CSE 的 _x_ 冲突）
    
    注意：
    - 必须在语法转换之后进行重命名
    - 保留字转换后的变量需要在 drop_symbols 时排除
    """

    def __init__(self, funcs_map, targets_map, args_map=None):

        if args_map is None:
            # 保留字
            args_map = {'True': "_TRUE_", 'False': "_FALSE_", 'None': "_NONE_"}
        self.funcs_old = set()
        self.args_old = set()
        self.targets_old = set()
        self.funcs_new = set()
        self.args_new = set()
        self.targets_new = set()
        # 映射
        self.funcs_map = funcs_map
        # 由于None等常量无法在sympy中正确处理，只能改成Symbol变量
        # !!!一定要在drop_symbols时排除
        self.args_map = args_map
        # 只对非下划线开头的生效
        self.targets_map = targets_map

    def visit_Call(self, node):
        # 提取函数名
        self.funcs_old.add(node.func.id)
        node.func.id = self.funcs_map.get(node.func.id, node.func.id)
        self.funcs_new.add(node.func.id)
        # 提取参数名
        for i, arg in enumerate(node.args):
            if isinstance(arg, ast.Name):
                self.args_old.add(arg.id)
                arg.id = self.args_map.get(arg.id, arg.id)
                self.args_new.add(arg.id)
            if isinstance(arg, ast.Constant):
                old_arg_value = str(arg.value)
                if old_arg_value in self.args_map:
                    new_arg_value = self.args_map.get(old_arg_value, old_arg_value)
                    self.args_old.add(old_arg_value)
                    node.args[i] = ast.Name(new_arg_value, ctx=ast.Load())
                    self.args_new.add(new_arg_value)

        self.generic_visit(node)
        return node

    def __visit_Assign(self, target: expr):
        """处理赋值目标（赋值语句左边）的重命名
        
        高级用法：
        1. 下划线开头的变量自动添加序号，避免与 CSE 的 _x_ 冲突
        2. 记录所有新旧变量名到 targets_old/targets_new，用于后续处理
        3. 常量值也可以被重命名（如 1 -> ONE）
        
        重命名示例：
        - _A -> _A_0_ （第一次出现，自动序号）
        - _A -> _A_1_ （第二次出现，自动序号）
        - A -> B （通过 targets_map={'A': 'B'}）
        
        注意：
        - 下划线开头的变量：targets_map 不生效（见第 465-468 行），自动添加序号
        - 普通变量：通过 targets_map 映射重命名（见第 460 行）
        - 修改会同步到 args_map（见第 474 行），确保赋值右边引用也被替换
        
        为什么同步到 args_map：
        - 赋值右边可能引用左边重命名前的变量，需要同步替换
        - 例如：A = A+A，如果 A -> B，需要变成 B = B+B
        - args_map 用于在 visit_Assign 的后续步骤中替换右边的变量引用
        """
        old_target_id = target.id
        # 尝试从 targets_map 获取映射，如果没有则保持原名
        new_target_id = self.targets_map.get(old_target_id, old_target_id)
        self.targets_old.add(old_target_id)

        # 赋值给下划线开头代码时，对其进行重命名，方便重复书写表达式时不冲突
        # 注意：targets_map 对下划线开头的变量不生效，走自动序号逻辑
        # 原因：第 416 行注释说明 "只对非下划线开头的生效"
        if old_target_id.startswith('_'):
            # 减少与cse中_x_冲突，自动添加序号
            # 例如：_A -> _A_0_，_A -> _A_1_（第二次出现）
            # len(self.targets_new) 是当前已重命名的变量数量，用作序号
            new_target_id = f'{old_target_id}_{len(self.targets_new)}_'

        if old_target_id != new_target_id:
            self.targets_new.add(new_target_id)
            target.id = new_target_id
            # 记录修改的变量名，同步到 args_map，确保赋值右边引用也被替换
            # 这样在 visit_Assign 处理右边时，会用 args_map 替换变量名
            self.args_map[old_target_id] = new_target_id

        # 处理常量值的情况（如 1 -> ONE）
        if isinstance(target, ast.Constant):
            old_target_value = str(target.value)
            if old_target_value in self.args_map:
                new_target_value = self.args_map.get(old_target_value, old_target_value)
                self.args_old.add(old_target_value)
                target = ast.Name(new_target_value, ctx=ast.Load())
                self.args_new.add(new_target_value)

        return target

    def visit_Assign(self, node):
        """处理赋值语句的完整重命名流程
        
        处理步骤：
        1. 先递归处理子节点（generic_visit），确保右边表达式先被处理
        2. 重命名赋值目标（左边），调用 __visit_Assign
        3. 重命名赋值值（右边），使用 args_map 替换变量引用
        
        支持循环赋值：
        - _A = _A+1 调整成 _A_0_ = _A + 1
        - 通过先处理右边再处理左边，实现循环依赖的重命名
        
        元组解包：
        - A, B = func() 会分别重命名 A 和 B
        - target.elts 是元组的元素列表
        
        为什么先 generic_visit：
        - 确保右边的表达式先被处理（如函数调用、嵌套表达式）
        - 这样在处理左边时，右边的变量引用已经被正确处理
        """
        # 先递归处理子节点，确保右边表达式先被处理
        self.generic_visit(node)

        # 提取并重命名输出变量名（赋值目标）
        for i, target in enumerate(node.targets):
            if isinstance(target, ast.Tuple):
                # 元组解包：A, B = func()
                # target.elts 是元组的元素列表，如 [A, B]
                for j, t in enumerate(target.elts):
                    target.elts[j] = self.__visit_Assign(t)
            else:
                # 普通赋值：A = 1
                node.targets[i] = self.__visit_Assign(target)

        # 处理赋值右边的变量引用
        # 如果右边是变量名，用 args_map 替换
        # 例如：A = B，如果 B -> C，变成 A = C
        if isinstance(node.value, ast.Name):
            self.args_old.add(node.value.id)
            node.value.id = self.args_map.get(node.value.id, node.value.id)
            self.args_new.add(node.value.id)
        # 如果右边是常量，也可能需要替换（如 1 -> ONE）
        if isinstance(node.value, ast.Constant):
            old_node_value = str(node.value.value)
            if old_node_value in self.args_map:
                new_node_value = self.args_map.get(old_node_value, old_node_value)
                self.args_old.add(old_node_value)
                node.value = ast.Name(new_node_value, ctx=ast.Load())
                self.args_new.add(new_node_value)

        return node

    def visit_Compare(self, node):
        # 比较符的左右也可能是变量，要处理
        if isinstance(node.left, ast.Name):
            self.args_old.add(node.left.id)
            node.left.id = self.args_map.get(node.left.id, node.left.id)
            self.args_new.add(node.left.id)

        for i, com in enumerate(node.comparators):
            if isinstance(com, ast.Name):
                self.args_old.add(com.id)
                com.id = self.args_map.get(com.id, com.id)
                self.args_new.add(com.id)
            if isinstance(com, ast.Constant):
                old_com_value = str(com.value)
                if old_com_value in self.args_map:
                    new_com_value = self.args_map.get(old_com_value, old_com_value)
                    self.args_old.add(old_com_value)
                    node.comparators[i] = ast.Name(new_com_value, ctx=ast.Load())
                    self.args_new.add(new_com_value)

        self.generic_visit(node)
        return node

    def visit_IfExp(self, node):
        if isinstance(node.body, ast.Name):
            self.args_old.add(node.body.id)
            node.body.id = self.args_map.get(node.body.id, node.body.id)
            self.args_new.add(node.body.id)
        if isinstance(node.orelse, ast.Name):
            self.args_old.add(node.orelse.id)
            node.orelse.id = self.args_map.get(node.orelse.id, node.orelse.id)
            self.args_new.add(node.orelse.id)

        self.generic_visit(node)
        return node

    def visit_BinOp(self, node):
        if isinstance(node.left, ast.Name):
            self.args_old.add(node.left.id)
            node.left.id = self.args_map.get(node.left.id, node.left.id)
            self.args_new.add(node.left.id)
        if isinstance(node.right, ast.Name):
            self.args_old.add(node.right.id)
            node.right.id = self.args_map.get(node.right.id, node.right.id)
            self.args_new.add(node.right.id)
        if isinstance(node.left, ast.Constant):
            old_node_value = str(node.left.value)
            if old_node_value in self.args_map:
                new_node_value = self.args_map.get(old_node_value, old_node_value)
                self.args_old.add(old_node_value)
                node.left = ast.Name(new_node_value, ctx=ast.Load())
                self.args_new.add(new_node_value)
        if isinstance(node.right, ast.Constant):
            old_node_value = str(node.right.value)
            if old_node_value in self.args_map:
                new_node_value = self.args_map.get(old_node_value, old_node_value)
                self.args_old.add(old_node_value)
                node.right = ast.Name(new_node_value, ctx=ast.Load())
                self.args_new.add(new_node_value)

        self.generic_visit(node)
        return node

    def visit_UnaryOp(self, node):
        # -x
        if isinstance(node.operand, ast.Name):
            self.args_old.add(node.operand.id)
            node.operand.id = self.args_map.get(node.operand.id, node.operand.id)
            self.args_new.add(node.operand.id)
        if isinstance(node.operand, ast.Constant):
            old_operand_value = str(node.operand.value)
            if old_operand_value in self.args_map:
                new_operand_value = self.args_map.get(old_operand_value, old_operand_value)
                self.args_old.add(old_operand_value)
                node.operand = ast.Name(new_operand_value, ctx=ast.Load())
                self.args_new.add(new_operand_value)

        self.generic_visit(node)
        return node

    def visit_Subscript(self, node):
        self.args_old.add(node.value.id)
        node.value.id = self.args_map.get(node.value.id, node.value.id)
        self.args_new.add(node.value.id)

        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node):
        return node

    def visit_ClassDef(self, node):
        return node

    def visit_Import(self, node):
        return node

    def visit_ImportFrom(self, node):
        return node


class KeywordToPositionalTransformer(ast.NodeTransformer):
    """将关键字参数转换为位置参数
    
    高级用法：
    1. inspect.signature 获取函数签名
    2. 按参数顺序重新排列参数
    3. 自动填充默认值
    
    应用场景：
    - 某些代码生成目标不支持关键字参数
    - 需要将 func(a=1, b=2) 转换为 func(1, 2)
    
    Parameters
    ----------
    function_mapping : dict
        函数名到实际函数对象的映射
    
    注意：
    - 需要实际的函数对象来获取签名
    - 转换失败时返回原节点
    """
    def __init__(self, function_mapping):
        self.function_mapping = function_mapping  # 函数名到实际函数的映射

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.keywords:
            func_name = node.func.id
            if func_name in self.function_mapping:
                return self.transform_call(node, self.function_mapping[func_name])
        return node

    def transform_call(self, node, target_func):
        try:
            # 获取函数参数签名
            sig = inspect.signature(target_func)
            param_names = list(sig.parameters.keys())

            # 构建参数映射
            arg_mapping = {}
            # 处理现有位置参数
            for i, arg in enumerate(node.args):
                if i < len(param_names):
                    arg_mapping[param_names[i]] = arg

            # 处理关键字参数
            for keyword in node.keywords:
                if keyword.arg in param_names:
                    arg_mapping[keyword.arg] = keyword.value

            # 按参数顺序构建新的位置参数列表
            new_args = []
            for param_name in param_names:
                if param_name in arg_mapping:
                    new_args.append(arg_mapping[param_name])
                else:
                    # 对于没有提供的参数，需要处理默认值
                    param = sig.parameters[param_name]
                    if param.default != inspect.Parameter.empty:
                        # 使用默认值
                        new_args.append(ast.Constant(value=param.default))
                    else:
                        # 必需参数缺失，保持原样或报错
                        return node

            # 创建新的调用节点
            new_node = ast.Call(
                func=node.func,
                args=new_args,
                keywords=[]
            )
            return new_node

        except Exception as e:
            # 转换失败时返回原节点
            return node


def source_replace(source: str) -> str:
    """预处理源代码，转换自定义语法
    
    高级用法：
    1. 正则表达式替换三元表达式
    2. 使用 while 循环处理嵌套的三元表达式
    3. 使用 black 格式化代码
    
    转换规则：
    - C?T:F -> C or True if( T )else F
    - || -> |
    - && -> &
    
    注意：
    - 三元表达式的转换是临时的，需要在 Transformer 中修正
    - 使用 or True 标记需要反转的条件
    - re.S 标志使 . 匹配换行符
    """
    # 三元表达式转换成 错误版if( )else，一定得在Transformer中修正
    num = 1
    while num > 0:
        # 利用or 的优先级最低，构造特殊的if else，只要出现，就认为位置要替换
        # C?T:F --> C or True if( T )else F
        source, num = re.subn(r'\?(.+?):(.+?)', r' or True if( \1 )else \2', source, flags=re.S)
        # break
    # 或、与
    source = source.replace('||', '|').replace('&&', '&')
    # IndentationError: unexpected indent
    # 嵌套函数前有空格，会报错
    source = format_str(source, mode=Mode(line_length=600, magic_trailing_comma=True))
    return source


def assigns_to_dict(assigns):
    """赋值表达式转成字典"""
    return {ast.unparse(a.targets): ast.unparse(a.value) for a in assigns}


def assigns_to_list(assigns):
    """赋值表达式转成列表"""
    outputs = []
    for i, a in enumerate(assigns):
        comment = "#"
        if i + 1 < len(assigns):
            b = assigns[i + 1]
            if isinstance(b, ast_comments.Comment):
                # comment = ast_comments.unparse(b)
                comment = b.value
        if isinstance(a, ast.Assign):
            outputs.append((ast.unparse(a.targets), ast.unparse(a.value), comment))
    return outputs


def raw_to_code(raw):
    """导入语句转字符列表"""
    return '\n'.join([ast_comments.unparse(a) for a in raw])


def sources_to_asts(*sources, convert_xor: bool, function_mapping):
    """将多份源代码转换为 AST
    
    高级用法：
    1. 支持函数定义或直接代码块
    2. 使用 Transformer 链进行多次转换
    3. 分离赋值语句和其他语句
    
    转换流程：
    1. source_replace 预处理
    2. SyntaxTransformer 语法转换
    3. KeywordToPositionalTransformer 参数转换
    4. RenameTransformer 变量重命名
    
    Parameters
    ----------
    *sources : str
        多份源代码字符串
    convert_xor : bool
        ^ 转换为乘方还是异或
    function_mapping : dict
        函数名到函数对象的映射
    
    Returns
    -------
    tuple
        (raw_code, assigns, funcs_new, args_new, targets_new)
    """
    def _source_to_asts(source):
        """源代码"""
        tree = ast_comments.parse(source_replace(source))

        if isinstance(tree.body[0], ast.FunctionDef):
            body = tree.body[0].body
        else:
            body = tree.body

        return body

    tree = ast_comments.parse("")
    for arg in sources:
        tree.body.extend(_source_to_asts(arg))

    # Transformer 链：按顺序应用多个转换器
    t1 = SyntaxTransformer(convert_xor)
    t1.visit(tree)
    t2 = KeywordToPositionalTransformer(function_mapping)
    t2.visit(tree)
    t = RenameTransformer({}, {})
    t.visit(tree)

    raw = []
    assigns = []

    for i, node in enumerate(tree.body):
        # 特殊处理的节点
        if isinstance(node, ast.Assign):
            assigns.append(node)
            continue
        # TODO 是否要把其它语句也加入？是否有安全问题？
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
            raw.append(node)
            continue
        if isinstance(node, ast_comments.Comment):
            # 添加注释
            if node.inline and isinstance(tree.body[i - 1], ast.Assign):
                assigns.append(node)
                continue

    return raw_to_code(raw), assigns_to_list(assigns), t.funcs_new, t.args_new, t.targets_new


def _add_default_type(globals_):
    """添加默认的 sympy 类型到全局命名空间
    
    高级用法：
    1. 自动注册常用的 sympy 类型
    2. 省去用户手动导入的工作
    3. 支持在表达式中直接使用 Add、Mul 等
    
    注册的类型：
    - Add: 加法
    - Mul: 乘法
    - Pow: 乘方
    - Eq: 等于
    - Not: 非
    - Xor: 异或
    
    Parameters
    ----------
    globals_ : dict
        全局命名空间字典
    
    Returns
    -------
    dict
        更新后的全局命名空间
    """
    # 这种写法可以省去由用户导入Eq一类的工作
    globals_['Add'] = Add
    globals_['Mul'] = Mul
    globals_['Pow'] = Pow
    globals_['Eq'] = Eq
    globals_['Not'] = Not
    globals_['Xor'] = Xor
    return globals_


def sources_to_exprs(globals_, *sources, convert_xor: bool):
    """将源代码转换为 sympy 表达式
    
    高级用法：
    1. 提取全局命名空间中的函数
    2. 添加默认的 sympy 类型
    3. 注册所有符号到 sympy
    
    转换流程：
    1. 提取函数映射
    2. 添加默认类型
    3. 转换为 AST
    4. 注册符号
    5. 转换为表达式
    
    Parameters
    ----------
    globals_ : dict
        全局命名空间
    *sources : str
        源代码字符串
    convert_xor : bool
        ^ 转换为乘方还是异或
    
    Returns
    -------
    tuple
        (raw_code, exprs_list)
        - raw_code: 导入语句等非表达式代码
        - exprs_list: 表达式列表
    """
    # 提取函数映射，用于关键字参数转换
    function_mapping = {k: v for k, v in globals_.items() if inspect.isfunction(v)}
    # 添加默认的 sympy 类型
    globals_ = _add_default_type(globals_)

    raw, assigns, funcs_new, args_new, targets_new = sources_to_asts(*sources, convert_xor=convert_xor, function_mapping=function_mapping)
    # 支持OPEN[1]转ts_delay(OPEN,1)
    funcs_new.add('ts_delay')

    # 注册符号到 sympy
    register_symbols(funcs_new, globals_, is_function=True)
    register_symbols(args_new, globals_, is_function=False)
    register_symbols(targets_new, globals_, is_function=False)
    return raw, list_to_exprs(assigns, globals_)
