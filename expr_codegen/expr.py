from functools import reduce

from sympy import Mul, preorder_traversal, symbols, Function, simplify, Add, Basic, Symbol, sympify, FunctionClass
from loguru import logger

# 预定义前缀，算子用前缀进行区分更方便。
# 当然也可以用是否在某容器中进行分类
CL = 'cl'  # 列算子, column
TS = 'ts'  # 时序算子, time-series
CS = 'cs'  # 横截面算子 cross section
GP = 'gp'  # 分组算子。group 分组越小，速度越慢

CL_TUP = (CL,)  # 整列元组
CL_SET = {CL_TUP}  # 整列集合


def is_symbol(x, globals_):
    """判断 x 是否已经是 sympy 中定义的符号
    
    高级用法：
    1. 检查 Symbol 类型（如 OPEN）
    2. 检查类型本身（如 Eq），使用 type(s) is type 而非 isinstance
    3. 检查 FunctionClass（如 Not），这是 sympy 中未调用的函数类
    
    Parameters
    ----------
    x : str
        符号名称
    globals_ : dict
        全局命名空间字典
    
    Returns
    -------
    bool
        是否已定义
    """
    s = globals_.get(x, None)
    if s is None:
        return False
    if isinstance(s, Symbol):
        # OPEN 等列名符号
        return True
    if type(s) is type:
        # Eq 等类型本身，注意这里用 type(s) is type 而非 isinstance
        # 因为 isinstance(Eq, type) 返回 True，但我们想判断 s 本身是否是类型
        return issubclass(s, Basic)
    if isinstance(s, FunctionClass):
        # Not 等函数类，FunctionClass 是 sympy 中未调用的函数的类型
        # 例如 Not 是 FunctionClass，而 Not(x) 是一个 Not 类型的表达式
        return True
    return False


def register_symbols(syms, globals_, is_function: bool):
    """将字符串符号注册到 sympy 的全局命名空间中
    
    高级用法：
    1. 使用 symbols() 批量创建 sympy 符号
    2. cls=Function 创建函数符号，cls=Symbol 创建普通符号
    3. seq=True 返回列表而非元组
    4. 函数符号必须使用 globals().copy()，因为注册后不能再调用
    
    Parameters
    ----------
    syms : list of str
        要注册的符号名称列表
    globals_ : dict
        全局命名空间字典
    is_function : bool
        是否为函数符号（如 ts_mean）
    
    Returns
    -------
    dict
        更新后的全局命名空间
    
    注意：
    函数符号注册后不能再调用，所以一定要用 globals().copy() 创建副本
    否则会影响后续的符号注册和表达式解析
    """
    # Eq等已经是sympy的符号不需注册
    syms = [s for s in syms if not is_symbol(s, globals_)]
    if len(syms) == 0:
        return globals_

    if is_function:
        # 函数被注册后不能再调用，所以一定要用globals().copy()
        # 这是因为 Function 类型的符号注册后，sympify 会将其视为已定义的函数
        # 如果不使用副本，会影响后续的符号注册
        syms = symbols(','.join(syms), cls=Function, seq=True)
    else:
        syms = symbols(','.join(syms), cls=Symbol, seq=True)
    syms = {s.name: s for s in syms}
    globals_.update(syms)
    return globals_


def list_to_exprs(exprs_src, globals_):
    """将字符串表达式列表转换为 sympy 表达式对象
    
    高级用法：
    1. sympify 将字符串转换为 sympy 表达式
    2. evaluate=False 非常重要，阻止 sympy 自动化简表达式
    3. 例如 evaluate=True 时，1+1 会变成 2，而我们希望保持 1+1 的形式
    
    Parameters
    ----------
    exprs_src : list of tuple
        (变量名, 表达式字符串, 注释) 的列表
    globals_ : dict
        全局命名空间，包含已注册的符号
    
    Returns
    -------
    list of tuple
        (变量名, sympy 表达式对象, 注释) 的列表
    """
    return [(k, sympify(v, globals_, evaluate=False), c) for k, v, c in exprs_src]


def append_node(node, output_exprs):
    """添加到队列。其中，-x将转为x

    此举是为了防止公共表达式中出现大量-x这种情况

    Parameters
    ----------
    node
        表达式当前节点
    output_exprs
        表达式列表

    Returns
    -------
    表达式列表

    """
    if isinstance(node, Mul):
        if node.args[0] == -1 or node.args[0] == 1:
            # 可能是-1也可能是1.0
            for arg in node.args[1:]:
                if arg.is_Atom:
                    continue
                output_exprs.append(arg)
                # print(1, arg)
        else:
            output_exprs.append(node)
            # print(2, node)
    else:
        output_exprs.append(node)
        # print(3, node)

    return output_exprs


def get_symbols(expr, syms=None, return_str=True):
    """得到"""
    if syms is None:
        syms = []

    for arg in expr.args:
        if arg.is_Symbol:
            if return_str:
                syms.append(arg.name)
            else:
                syms.append(arg)
        elif arg.is_Number:
            # alpha_001 = log(1)+1
            if return_str:
                syms.append(str(arg))
            else:
                syms.append(arg)
        else:
            get_symbols(arg, syms, return_str)
    return syms


def is_NegativeX(expr):
    """-x, 但-ts_sum格式不返回False"""
    if isinstance(expr, Mul):
        if expr.args[0] == -1 and len(expr.args) == 2 and expr.args[1].is_Atom:
            return True
    return False


def is_simple_expr(expr):
    if isinstance(expr, Mul):
        if expr.args[0] == -1 and len(expr.args) == 2 and expr.args[1].is_Atom:
            return True
    if isinstance(expr, Symbol):
        return True
    return False


def get_current_by_prefix(expr, date, asset, **kwargs):
    """表达式根节点信息。按名称前缀。例如

    OPEN取的是OPEN，得cl
    ts_mean取的ts_mean,得ts
    -ts_mean取的是-,得cl
    """
    if expr.is_Function:
        if hasattr(expr, 'name'):  # Or 没有名字
            prefix1 = expr.name[2]
            if prefix1 == '_':
                prefix2 = expr.name[:2]

                if prefix2 == TS:
                    return TS, asset
                if prefix2 == CS:
                    return CS, date
                if prefix2 == GP:
                    return GP, date, expr.args[0].name
    # 不需分组
    return CL_TUP


def get_current_by_name(expr, ts_names, cs_names, gp_names, date, asset, **kwargs):
    """表达式根节点信息。按名称。

    Parameters
    ----------
    expr
    ts_names
        时序算子名称字符串集合
    cs_names
        横截面算子名称字符串集合
    gp_names
        分组算子名称字符串集合
    date
        日期字符串
    asset
        资产字符串
    kwargs

    """
    if expr.is_Function:
        if hasattr(expr, 'name'):  # Or 没有名字
            if expr.name in ts_names:
                return TS, asset
            if expr.name in cs_names:
                return CS, date
            if expr.name in gp_names:
                return GP, date, expr.args[0].name

    # 不需分组
    return CL_TUP


# 调试用，勿删
# __level__ = 0


def get_children(func, func_kwargs, expr, output_exprs, output_symbols, date, asset):
    """递归遍历表达式树，提取需要单独处理的子表达式
    
    高级用法：
    1. 递归遍历表达式树，获取每个节点的分类信息
    2. 使用 reduce 进行集合合并操作，lambda x, y: x | y 表示集合并集
    3. 根据当前节点和子节点的分类差异，决定是否提取子表达式
    4. 核心逻辑：如果外部分类与内部分类不同，需要提取子表达式单独处理
    
    例如：
    -ts_corr 返回 {ts} 而不是 {cl}
    -ts_corr+cs_rank 返回 {ts,cs}
    -OPEN-CLOSE 返回 {cl}
    
    Parameters
    ----------
    func : callable
        表达式根分类函数（如 get_current_by_prefix）
    func_kwargs : dict
        func 对应的参数字典
    expr : sympy.Expr
        当前表达式节点
    output_exprs : list
        输出分割后的子表达式列表
    output_symbols : list
        输出每个子表达式中的符号列表
    date : str
        日期字段名
    asset : str
        资产字段名
    
    Returns
    -------
    set
        当前节点包含的分类集合（如 {('ts', 'asset'), ('cs', 'date')})
    
    核心算法：
    1. 递归获取所有子节点的分类信息
    2. 使用 reduce 合并所有子节点的分类集合
    3. 如果子节点分类不统一（>=2种），提取所有非原子子表达式
    4. 如果当前分类与子分类不同，提取不匹配的子表达式
    5. 返回当前节点的分类信息供父节点使用
    """
    # global __level__
    # __level__ += 1

    try:
        # 获取当前节点的分类信息
        curr = func(expr, date, asset, **func_kwargs)
        # 递归获取所有子节点的分类信息
        children = [get_children(func, func_kwargs, a, output_exprs, output_symbols, date, asset) for a in expr.args]

        # print(expr, curr, children, __level__)
        # if __level__ == 6:
        #     print(expr, curr, children)

        # 多个集合合成一个去重
        # reduce(lambda x, y: x | y, children, set()) 表示将 children 中的所有集合进行并集操作
        # 初始值为 set()，逐步合并每个子集合
        unique = reduce(lambda x, y: x | y, children, set()) - CL_SET

        if len(unique) >= 2:
            # 大于1，表示内部不统一，内部都要处理
            # 例如 ts_corr + cs_rank，内部既有 ts 又有 cs，需要分别处理
            for i, child in enumerate(children):
                # alpha_047无法正确输出
                if expr.args[i].is_Atom:
                    # print(expr.args[i], 'is_Atom 1')
                    continue
                output_exprs = append_node(expr.args[i], output_exprs)
        elif curr[0] != CL:
            # 外部与内部不同，需处理
            # 例如 -ts_corr，外部是 cl，内部是 ts，需要提取 ts_corr 单独处理
            for i, child in enumerate(children):
                # 不在子中即表示不同
                if curr in child:
                    continue
                if expr.args[i].is_Atom:
                    # print(expr.args[i], 'is_Atom 2')
                    continue
                output_exprs = append_node(expr.args[i], output_exprs)
        else:
            # ts_sum(OPEN, 5)*ts_sum(RETURNS, 5) ('cl',) [{('ts', 'asset', 'date')}, {('ts', 'asset', 'date')}] 6 alpha_008
            # 外部是 cl，内部都是 ts，不需要提取，因为可以在同一个 groupby 中处理
            pass
            # if isinstance(expr, Mul):
            #     output_exprs = append_node(expr, output_exprs)

        # 按需返回，当前是基础算子就返回下一层信息，否则返回当前
        if curr[0] == CL:
            if expr.is_Symbol:
                # 汇总符号列表
                output_symbols.append(expr)
            # 返回子中出现过的集合{ts cs gp}
            return unique
        else:
            # 当前算子，非列算子，直接返回，如{ts} {cs} {gp}
            return {curr}
    finally:
        # __level__ -= 1
        pass


def get_key(children):
    """根据子表达式的分类信息生成字典键
    
    高级用法：
    1. 此函数只能在先抽取出子表达式后再 CSE，然后才能调用
    2. 如果子表达式分类不统一（>=2种），说明之前的提取不完整，会报错
    3. 这是 DAG 分组的关键，确保每个表达式只有一种正确的分组方式
    
    重要限制：
    必须在公共子表达式消除（CSE）之后调用，否则会报错
    因为 CSE 会将重复的表达式替换为变量，确保分类统一
    
    Parameters
    ----------
    children : set
        子表达式的分类集合，如 {('ts', 'asset'), ('cs', 'date')}
    
    Returns
    -------
    tuple
        用于字典的键，表示分组方式
        - ('cl',): 列算子，不需要分组
        - ('ts', 'asset'): 时序算子，按资产分组
        - ('cs', 'date'): 截面算子，按日期分组
        - ('gp', 'date', 'group'): 分组算子，按日期和分组字段分组
    
    Raises
    ------
    AssertionError
        如果子表达式分类不统一，说明之前的提取不完整
    """
    if len(children) == 0:
        # OPEN等因子会走这一步，没有子表达式，属于列算子
        return CL_TUP
    elif len(children) == 1:
        # 只有一种分法，最合适的方法
        # 例如只有 ts 或只有 cs，可以直接使用该分类
        return list(children)[0]
    else:
        # 如果有多种分类，说明之前的子表达式提取不完整
        # 例如既有 ts 又有 cs，应该在 get_children 中就提取出来
        assert False, f'{children} 无法正确分类，之前没有分清'


def replace_exprs(exprs):
    """使用替换的方式简化表达式"""
    # Alpha101中大量ts_sum(x, 10)/10, 转成ts_mean(x, 10)
    exprs = [(k, _replace__ts_sum__to__ts_mean(v), c) for k, v, c in exprs]
    # alpha_031中大量cs_rank(cs_rank(x)) 转成cs_rank(x)
    exprs = [(k, _replace__repeat(v), c) for k, v, c in exprs]
    # 1.0*VWAP转VWAP
    exprs = [(k, _replace__one_mul(v), c) for k, v, c in exprs]
    # 将部分参数为1的ts函数进行简化
    exprs = [(k, _replace__ts_xxx_1(v), c) for k, v, c in exprs]
    # ts_delay转成ts_delta
    exprs = [(k, _replace__ts_delay__to__ts_delta(v), c) for k, v, c in exprs]

    return exprs


def get_node_name(node):
    """得到节点名"""
    if hasattr(node, 'name'):
        # 如 ts_arg_max
        node_name = node.name
    else:
        # 如 log
        node_name = str(node.func)
    return node_name


def _replace__ts_sum__to__ts_mean(e):
    """将 ts_sum(x, y)/y 转成 ts_mean(x, y)
    
    高级用法：
    1. preorder_traversal 是 sympy 的树遍历函数，按前序遍历表达式树
    2. is_Rational 判断是否为有理数（如 1/10）
    3. node.args[0].p 和 node.args[0].q 分别表示分子和分母
    4. xreplace 是 sympy 的表达式替换函数
    
    优化目的：
    Alpha101 中大量出现 ts_sum(x, 10)/10，转换为 ts_mean(x, 10) 更高效
    
    Parameters
    ----------
    e : sympy.Expr
        输入表达式
    
    Returns
    -------
    sympy.Expr
        优化后的表达式
    """
    if not isinstance(e, Basic):
        return e

    # TODO: 这里重新定义的ts_mean与外部已经定义好的是否同一个？
    # 注意：这里创建的 ts_mean 符号可能与外部定义的不是同一个对象
    # 但在 sympify 时会通过全局命名空间匹配，所以不影响使用
    ts_mean = symbols('ts_mean', cls=Function)

    replacements = []
    # preorder_traversal 按前序遍历表达式树，访问每个节点
    for node in preorder_traversal(e):
        # 检查是否为乘法，且第一个参数是有理数，第二个参数是函数
        if node.is_Mul and node.args[0].is_Rational and node.args[1].is_Function:
            node_name = get_node_name(node.args[1])
            if node_name == 'ts_sum':
                # 检查是否为 ts_sum(x, y)/y 的形式
                # node.args[0].p 是分子，node.args[0].q 是分母
                # node.args[1].args[1] 是 ts_sum 的第二个参数（窗口期）
                if node.args[1].args[1] == node.args[0].q and node.args[0].p == 1:
                    replacements.append((node, ts_mean(node.args[1].args[0], node.args[1].args[1])))
    for node, replacement in replacements:
        logger.debug("{} -> {}", node, replacement)
        # xreplace 批量替换表达式中的节点
        e = e.xreplace({node: replacement})
    return e


def _replace__repeat(e):
    """cs_rank(cs_rank(x)) 转成 cs_rank(x)
    sign(sign(x)) 转成 sign(x)
    Abs(Abs(x)) 转成 Abs(x)
    """
    if not isinstance(e, Basic):
        return e

    replacements = []
    for node in preorder_traversal(e):
        # print(node)
        if len(node.args) == 0:
            continue
        node_name = get_node_name(node)
        node_args0_name = get_node_name(node.args[0])
        if node_name == node_args0_name:
            if node_name in ('cs_rank', 'sign', 'Abs', 'abs_'):
                replacements.append((node, node.args[0]))
    for node, replacement in replacements:
        logger.debug("{} -> {}", node, replacement)
        e = e.xreplace({node: replacement})
    return e


def _replace__one_mul(e):
    """1.0*VWAP转成VWAP"""
    if not isinstance(e, Basic):
        return e

    replacements = []
    for node in preorder_traversal(e):
        # print(node)
        if isinstance(node, Mul) and node.args[0] == 1:
            if len(node.args) > 2:
                replacements.append((node, Mul._from_args(node.args[1:])))
            else:
                replacements.append((node, node.args[1]))
    for node, replacement in replacements:
        logger.debug("{} -> {}", node, replacement)
        e = e.xreplace({node: replacement})
    return e


def _replace__ts_xxx_1(e):
    """ts_xxx部分函数如果参数为1，可直接丢弃"""
    if not isinstance(e, Basic):
        return e

    replacements = []
    for node in preorder_traversal(e):
        node_name = get_node_name(node)
        if node_name in ('ts_mean', 'ts_sum', 'ts_decay_linear',
                         'ts_max', 'ts_min', 'ts_arg_max', 'ts_arg_min',
                         'ts_product', 'ts_std_dev', 'ts_rank'):
            try:
                if node.args[1] <= 1:
                    replacements.append((node, node.args[0]))
            except Exception as ex:
                logger.warning("{} | {}: {}", e, node_name, ex)
    for node, replacement in replacements:
        logger.debug("{} -> {}", node, replacement)
        e = e.xreplace({node: replacement})
    return e


def _replace__ts_delay__to__ts_delta(e):
    """将 -ts_delay(x, y) 转成 ts_delta(x, y) - x
    
    本质上为 x - ts_delay(x, y) 转成 ts_delta(x, y)
    
    高级用法：
    1. 识别 x - ts_delay(x, y) 的模式，转换为 ts_delta(x, y)
    2. 利用 sympy 的 simplify 自动化简，消除对冲的变量
    3. Add._from_args 从参数列表构造 Add 表达式
    
    优化目的：
    ts_delta 比 x - ts_delay 更高效，且可以减少计算量
    
    例如：
    OPEN - ts_delay(OPEN, 5) + (CLOSE - ts_delay(CLOSE, 5))
    结果：ts_delta(CLOSE, 5) + ts_delta(OPEN, 5)
    
    Parameters
    ----------
    e : sympy.Expr
        输入表达式
    
    Returns
    -------
    sympy.Expr
        优化后的表达式
    
    算法说明：
    1. 遍历所有加法节点
    2. 找到 -ts_delay(x, y) 的模式（即 arg.args[0] == -1 且 arg.args[1] 是 ts_delay）
    3. 替换为 ts_delta(x, y) - x
    4. 使用 simplify 化简，消除对冲的变量（如 x - x = 0）
    5. 只有化简后参数变少才认为替换成功
    """
    if not isinstance(e, Basic):
        return e

    ts_delta = symbols('ts_delta', cls=Function)

    replacements = []
    for node in preorder_traversal(e):
        if node.is_Add:
            new_args = []
            for arg in node.args:
                if arg.is_Mul:
                    # 检查是否为 -ts_delay(x, y) 的形式
                    if arg.args[0] == -1 and arg.args[1].is_Function and get_node_name(arg.args[1]) == 'ts_delay':
                        # 替换为 ts_delta(x, y) - x
                        # 添加 ts_delta(x, y)
                        new_args.append(ts_delta(arg.args[1].args[0], arg.args[1].args[1]))
                        # 添加 -x
                        new_args.append(-arg.args[1].args[0])
                    else:
                        new_args.append(arg)
                else:
                    new_args.append(arg)
            if len(new_args) > len(node.args):
                # 长度变长，表示成功实现了调整
                # 例如：OPEN - ts_delay(OPEN, 5) 变成 ts_delta(OPEN, 5) - OPEN
                tmp_args = simplify(Add._from_args(new_args))
                # 优化后长度变短，表示有变量对冲掉了，成功
                # 例如：ts_delta(OPEN, 5) - OPEN + OPEN 化简后只剩 ts_delta(OPEN, 5)
                if len(tmp_args.args) < len(new_args):
                    replacements.append((node, tmp_args))
    for node, replacement in replacements:
        logger.debug("{} -> {}", node, replacement)
        e = e.xreplace({node: replacement})
    return e
