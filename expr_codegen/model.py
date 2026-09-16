from functools import reduce
from itertools import product, permutations

import networkx as nx
from sympy import symbols

from expr_codegen.dag import zero_indegree, hierarchy_pos, remove_paths_by_zero_outdegree
from expr_codegen.expr import CL, get_symbols, get_children, get_key, is_simple_expr

_RESERVED_WORD_ = {'_NONE_', '_TRUE_', '_FALSE_'}


class ListDictList:
    """嵌套列表数据结构，用于组织分层的表达式
    
    高级用法：
    1. 三层嵌套结构：列表 -> 字典 -> 列表
    2. 用于存储按分组方式（ts/cs/gp）组织的表达式
    3. 支持动态添加、合并、优化操作
    
    数据结构示例：
    [
        {'ts': [1, 2], 'cs': [2], 'gp_date': [2], 'gp_key': [2], }
        {'ts': [1], 'cs': [1],}
    ]
    
    - 最外层列表：表示不同的计算层（按依赖关系分层）
    - 第二层字典：同一层内按分组方式（ts/cs/gp）分类
    - 第三层列表：同一分组方式下的表达式列表
    """

    def __init__(self):
        self._list = []

    def clear(self):
        """清空"""
        self._list = []

    def values(self):
        return self._list

    def next_row(self):
        """移动到新的一行"""
        self._list.append({})

    def append(self, key, item):
        """自动放入同key的字典中"""
        last_row = self._list[-1]
        v = last_row.get(key, None)
        if v is None:
            # 同一行的新一列
            last_row[key] = [None, item]
        else:
            last_row[key].append(item)

    def filter_empty(self):
        """过滤空值"""
        new_list = []
        for row in self._list:
            try_del1 = []
            for k, v in row.items():
                if len(v) == 0:
                    try_del1.append(k)
            for k in try_del1:
                row.pop(k)
            if len(row) > 0:
                new_list.append(row)
        self._list = new_list

    def back_merge(self):
        """向上合并，将相同类型的表达式向前合并
        
        高级用法：
        1. 使用 reduce 函数扁平化嵌套结构
        2. reduce(lambda x, y: x + list(y.keys()), self._list, []) 将所有层的 keys 合并成一个列表
        3. 相邻的相同 key 合并，减少 groupby 次数
        
        优化目的：
        减少运行时的 groupby 次数，提高性能
        例如：两层都有 ts 分组，可以合并成一层
        """
        # reduce 将多层字典的 keys/values 合并成扁平列表
        keys = reduce(lambda x, y: x + list(y.keys()), self._list, [])
        values = reduce(lambda x, y: x + list(y.values()), self._list, [])

        new_keys = []
        new_values = []
        last_k = None
        last_v = None
        for k, v in zip(keys, values):
            # 当前是整列时可以向上合并，但前一个是gp_xxx一类时不合并，因为循环太多次了
            # if (last_v is not None) and (k[0] == CL) and (last_k[0] != GP):
            if (last_v is not None) and (k == last_k):
                # print(1, k, last_k)
                # 相同 key，合并到上一层
                last_v.extend(v)
                v.clear()
            else:
                # print(2, k, last_k)                
                # 不同 key，保留
                new_keys.append(k)
                new_values.append(v)
                last_v = v
                last_k = k

    def optimize(self, merge: bool):
        """将多组groupby根据规则进行合并，减少运行时间"""
        # 接龙。groupby的数量没少，首尾接龙数据比较整齐
        self._list = chain_create(self._list)
        if merge:
            # 首尾一样，接上去
            self.back_merge()
        # 出现了空行，删除
        self.filter_empty()

    def drop_symbols(self):
        """计算每个步骤之后可以删除的变量名
        
        高级用法：
        1. 反向遍历计算累积变量集合
        2. 使用集合差集计算不再需要的变量
        3. 用于内存优化，提前删除不再使用的中间变量
        
        算法说明：
        - l1: 每个表达式块使用的变量集合
        - l2: 从当前行到最后一行的累积变量集合（反向计算）
        - l3: 当前行之后不再出现的变量（可删除）
        
        例如：
        第1行使用 {A, B}
        第2行使用 {B, C}
        第3行使用 {C}
        
        反向累积：
        l2 = [{A,B,C}, {B,C}, {C}, {}]
        l3 = [{A}, {B}, {C}]
        
        表示第1行后可删除A，第2行后可删除B，第3行后可删除C
        """
        # 获取每一小块所用到的所有变量名
        l1 = []
        for row in self._list:
            for k, v in row.items():
                vv = []
                for v1 in v:
                    if v1 is None:
                        continue
                    vv.extend(v1[2])
                l1.append(set(vv))

        # 得到此行与之后都会出现的变量名
        # 反向遍历，计算累积集合
        l2 = [set()]
        s = set()
        for i in reversed(l1):
            s = s | i  # - {'_NONE_', '_TRUE_', '_FALSE_'}
            l2.append(s)
        l2 = list(reversed(l2))

        # 计算之后不会再出现的变量名
        # l2[i] - l2[i+1] 表示第i行之后不再出现的变量
        l3 = [list(s - e) for s, e in zip(l2[:-1], l2[1:])]

        return l3


def score1(row) -> int:
    # 首尾相连打分加1
    lst = [None] + [key for r in row for key in dict(r).keys()]
    return sum([x == y for x, y in zip(lst[:-1], lst[1:])])


def score2(row) -> float:
    # 最后一个ts越靠前，打分越高
    lst = ['ts'] + [key[0] for r in row for key in dict(r).keys()]
    return lst[::-1].index('ts') / len(lst)


def chain_create(nested_list):
    """接龙算法：优化多层表达式的执行顺序
    
    高级用法：
    1. itertools.permutations 生成每层的所有排列
    2. itertools.product 生成层与层之间的笛卡尔积
    3. 使用评分函数选择最优的执行顺序
    
    优化目的：
    减少数据在不同 groupby 之间的传递次数，提高性能
    
    测试用表达式：
    ma_10 = ts_mean(CLOSE, 10)
    MAMA_20 = ts_mean(ma_10, 20)
    alpha_031 = ((cs_rank(cs_rank(cs_rank(ts_decay_linear((-1 * cs_rank(cs_rank(ts_delta(CLOSE, 10)))), 10))))))
    
    算法说明：
    1. 对每一层的字典项生成所有排列（permutations）
    2. 生成层与层之间的笛卡尔积（product），得到所有可能的组合
    3. 对每种组合计算得分（score1 + score2）
    4. 选择得分最高的组合作为最优执行顺序
    
    评分标准：
    - score1: 首尾相连加分（相邻层使用相同分组方式）
    - score2: 最后一个 ts 靠前加分（时序计算优先）
    """
    perms = []
    for d in nested_list:
        # 每一层生成排列
        # permutations(d.items()) 生成字典项的所有排列
        perms.append(permutations(d.items()))

    last_score = float('-inf')
    last_row = None
    # 生成笛卡尔积
    # product(*perms) 生成所有层排列的组合
    for row in product(*perms):
        result = score1(row) + score2(row)
        # print(result, row)
        if result > last_score:
            last_score = result
            last_row = row

    return [dict(ro) for ro in last_row]


# ==========================

def create_dag_exprs(exprs):
    """根据表达式列表生成有向无环图（DAG）
    
    高级用法：
    1. 使用 networkx.DiGraph 创建有向无环图
    2. 节点表示表达式，边表示依赖关系
    3. zero_indegree 找到入度为0的节点（源节点）
    
    图结构说明：
    - 节点：每个表达式（包括中间变量和最终输出）
    - 边：从依赖节点指向被依赖节点
    - 例如：A = B + C，则有边 B->A 和 C->A
    
    Parameters
    ----------
    exprs : list
        表达式列表，每个元素为 (symbol, expr, comment)
    
    Returns
    -------
    nx.DiGraph
        表达式依赖关系的有向无环图
    """
    # 创建有向无环图
    G = nx.DiGraph()

    for symbol, expr, comment in exprs:
        # if symbol.name == 'GP_0':
        #     test = 1
        if expr.is_Symbol:
            # 如果表达式是符号（如 OPEN），直接添加边
            G.add_node(symbol.name, symbol=symbol, expr=expr, comment=comment)
            G.add_edge(expr.name, symbol.name)
        else:
            # 添加中间节点
            G.add_node(symbol.name, symbol=symbol, expr=expr, comment=comment)
            syms = get_symbols(expr, return_str=True)
            for sym in syms:
                # 由于边的原因，这里会主动生成一些源节点
                G.add_edge(sym, symbol.name)
            if len(syms) == 0:
                # GP_0033=log(1/2400) 这种常量表达式
                if hasattr(expr, 'name'):
                    G.add_edge(expr.name, symbol.name)
                else:
                    G.add_edge(str(expr), symbol.name)

    # 源始因子，添加属性
    # zero_indegree 找到入度为0的节点（没有依赖的节点）
    for node in zero_indegree(G):
        s = symbols(node)
        G.nodes[node]['symbol'] = s
        G.nodes[node]['expr'] = s
        G.nodes[node]['comment'] = "#"
    #
    # for node in zero_outdegree(G):
    #     print(11, G.nodes[node]['comment'])
    return G


def init_dag_exprs(G, func, func_kwargs, date, asset):
    """使用表达式信息初始化 DAG 节点属性
    
    高级用法：
    1. nx.topological_generations 按拓扑排序分层遍历图
    2. 为每个节点计算分类信息（children、key、symbols）
    3. generation 表示节点所在的层（依赖层级）
    
    拓扑排序说明：
    - 按依赖关系分层，先计算依赖节点，再计算被依赖节点
    - 例如：A = B + C, B = D，则分层为 [D], [B], [A]
    
    Parameters
    ----------
    G : nx.DiGraph
        有向无环图
    func : callable
        表达式分类函数
    func_kwargs : dict
        分类函数的参数
    date : str
        日期字段名
    asset : str
        资产字段名
    
    Returns
    -------
    nx.DiGraph
        更新后的图，每个节点包含 children、key、symbols、gen 属性
    """
    for i, generation in enumerate(nx.topological_generations(G)):
        # print(i, generation)
        for node in generation:
            expr = G.nodes[node]['expr']
            syms = []
            children = get_children(func, func_kwargs, expr, [], syms, date, asset)
            G.nodes[node]['children'] = children
            G.nodes[node]['key'] = get_key(children)
            G.nodes[node]['symbols'] = [str(s) for s in syms]
            G.nodes[node]['gen'] = i
            # print(G.nodes[node])
    return G


def merge_nodes_1(G: nx.DiGraph, keep_nodes, *args):
    """合并节点，从当前节点开始，查看是否可能替换前后两端的节点"""
    # 准备一个当前节点列表
    this_pred = args
    # 下一步不为空就继续
    while this_pred:
        next_pred = []
        for node in this_pred:
            if not G.has_node(node):
                continue
            pred = G.pred[node]
            if len(pred) == 0:
                # 到了最上层的因子，需停止
                continue
            dic = G.nodes[node]
            key = dic['key']
            expr = dic['expr']
            symbols = dic['symbols']
            if key[0] == CL:
                if is_simple_expr(expr):
                    # 检查表达式是否很简单, 是就替换，可能会替换多个
                    skip_expr_node(G, node, keep_nodes)
                else:
                    succ = G.succ[node]
                    # 下游只有一个，直接替换。
                    if len(succ) == 1:
                        for s in succ:
                            # if_else(_A>_B,_A,_B)会出现量次，不能删
                            if G.nodes[s]['symbols'].count(node) > 1:
                                continue
                            skip_expr_node(G, node, keep_nodes)
            else:
                # 复制一次，防止修改后报错
                for p in pred.copy():
                    # 在下游同一表达式中使用了多次，不替换
                    if symbols.count(p) > 1:
                        continue
                    d = G.nodes[p]
                    k = d['key']
                    e = d['expr']
                    if key == k:
                        # 同类型
                        succ = G.succ[p]
                        # 下游只有一个，直接替换。
                        if len(succ) == 1:
                            for s in succ:
                                if G.nodes[s]['symbols'].count(p) > 1:
                                    continue
                                skip_expr_node(G, p, keep_nodes)
            next_pred.extend(pred)
        # 更新下一次循环
        this_pred = list(set(next_pred))
    return G


def merge_nodes_2(G: nx.DiGraph, keep_nodes, *args):
    """合并节点，从当前节点开始，查看是否需要被替换，只做用于根节点"""
    # 准备一个当前节点列表
    this_pred = args
    # 下一步不为空就继续
    while this_pred:
        next_pred = []
        for node in this_pred:
            dic = G.nodes[node]
            expr = dic['expr']
            if not is_simple_expr(expr):
                continue
            pred = G.pred[node]
            for p in pred.copy():
                succ = G.succ[p]
                if len(succ) > 1:
                    # 上游节点只有一个下游，当前就是自己了
                    continue
                for s in succ:
                    if G.nodes[s]['symbols'].count(p) > 1:
                        continue
                    skip_expr_node(G, p, keep_nodes)
            # 只做根节点，所以没有下一次了
            # next_pred.extend(pred)
        # 更新下一次循环
        this_pred = list(set(next_pred))
    return G


def get_expr_labels(G, nodes=None):
    """得到表达式标签"""
    labels = {}
    if nodes is None:
        for n, d in G.nodes(data=True):
            labels[n] = '{symbol}={expr}'.format(**d)
    else:
        for n, d in G.nodes(data=True):
            if n not in nodes:
                continue
            labels[n] = '{symbol}={expr}'.format(**d)
    return labels


def draw_expr_tree(G: nx.DiGraph, root: str, ax=None):
    """画表达式树"""
    # 查找上游节点
    nodes = nx.ancestors(G, root) | {root}
    labels = get_expr_labels(G, nodes)
    # 子图
    view = nx.subgraph(G, nodes)
    # 位置
    pos = hierarchy_pos(G, root)
    nx.draw(view, ax=ax, pos=pos, labels=labels)


def skip_expr_node(G: nx.DiGraph, node, keep_nodes):
    """跳过中间节点，将两端的节点直接连接起来，同时更新表达式
    
    高级用法：
    1. 使用 xreplace 替换表达式中的符号
    2. 使用 product 生成所有前驱和后继的组合边
    3. 用于简化 DAG，减少不必要的中间节点
    
    优化场景：
    1. (A,B,C) 模式，直接成 (A,C)
       - B 是简单表达式（如 -A），可以直接内联
    2. (A,B,C), (D, B) 模式，变成 (A,C),(D,C)
       - B 被多个节点引用，但表达式简单，可以内联
    
    Parameters
    ----------
    G : nx.DiGraph
        有向无环图
    node : str
        要跳过的节点名
    keep_nodes : list
        需要保留的节点列表
    
    Returns
    -------
    nx.DiGraph
        更新后的图
    
    算法说明：
    1. 将当前节点的表达式内联到所有后继节点中
    2. 使用 product 生成所有前驱->后继的边
    3. 删除当前节点
    """
    if node in keep_nodes:
        return G

    pred = G.pred[node]
    succ = G.succ[node]
    if len(pred) == 0 or len(succ) == 0:
        return G

    # 取当前节点表达式
    d = G.nodes[node]
    expr = d['expr']
    symbol = d['symbol']

    for s in succ:
        e = G.nodes[s]['expr']
        # xreplace 将表达式中的 symbol 替换为 expr
        # 例如：symbol=_x_0, expr=-A，则 _x_0 + B 变成 -A + B
        e = e.xreplace({symbol: expr})
        G.nodes[s]['expr'] = e

    # 这里用了product生成多个关联边
    # product(pred, succ) 生成所有前驱和后继的组合
    # 例如 pred=[A,D], succ=[C]，则生成边 A->C, D->C
    G.add_edges_from(product(pred, succ))
    G.remove_node(node)
    return G


def dag_start(exprs_list, func, func_kwargs, date, asset):
    """初始生成DAG"""
    G = create_dag_exprs(exprs_list)
    G = init_dag_exprs(G, func, func_kwargs, date, asset)

    # 分层输出
    return G


def dag_middle(G, exprs_names, skip_columns, func, func_kwargs, date, asset, skip_simplify):
    """删除几个没有必要的节点"""
    # 以下划线开头的节点，不保留
    keep_nodes = [k for k in exprs_names if not k.startswith('_')]

    if not skip_simplify:
        # ts_rank(-RET - -RET, 20),防止替换成ts_rank(0, 20)
        G = merge_nodes_1(G, keep_nodes, *keep_nodes)
    G = merge_nodes_2(G, keep_nodes, *keep_nodes)

    # 移除0出度的节点，但保留部分
    G = remove_paths_by_zero_outdegree(G, set(keep_nodes) - set(skip_columns))

    # 由于表达式修改，需再次更新表达式
    G = init_dag_exprs(G, func, func_kwargs, date, asset)

    # 分层输出
    return G


def dag_end(G):
    """将 DAG 转换为 ListDictList 结构，用于代码生成
    
    高级用法：
    1. 按拓扑排序遍历 DAG，转换为分层结构
    2. 剔除第0层（基础因子如 CLOSE）
    3. 过滤保留字（_NONE_, _TRUE_, _FALSE_）
    
    转换流程：
    DAG -> ListDictList -> 代码生成
    
    Parameters
    ----------
    G : nx.DiGraph
        有向无环图
    
    Returns
    -------
    tuple
        (ListDictList, nx.DiGraph)
        - ListDictList: 分层的表达式结构
        - nx.DiGraph: 原始图
    """
    exprs_ldl = ListDictList()

    for i, generation in enumerate(nx.topological_generations(G)):
        exprs_ldl.next_row()
        for node in generation:
            key = G.nodes[node]['key']
            expr = G.nodes[node]['expr']
            comment = G.nodes[node]['comment']
            symbols = G.nodes[node]['symbols']
            # 这几个特殊的不算成字段名
            symbols = list(set(symbols) - _RESERVED_WORD_)

            exprs_ldl.append(key, (node, expr, symbols, comment))

    # 第0层是CLOSE等基础因子，剔除
    exprs_ldl._list = exprs_ldl.values()[1:]

    return exprs_ldl, G
