from itertools import product

import networkx as nx
# networkx 是 Python 中用于图论和网络分析的库
# DiGraph 表示有向图，支持各种图算法和可视化


def zero_indegree(G: nx.DiGraph):
    """获取入度为 0 的所有节点
    
    高级用法：
    1. G.in_degree() 返回所有节点的 (节点, 入度) 元组迭代器
    2. 入度为 0 表示没有依赖该节点的其他节点
    3. 在 DAG 中，入度为 0 的节点可以作为拓扑排序的起点
    
    Parameters
    ----------
    G : nx.DiGraph
        有向图
    
    Returns
    -------
    list
        入度为 0 的节点列表
    """
    return [v for v, d in G.in_degree() if d == 0]


def zero_outdegree(G: nx.DiGraph):
    """获取出度为 0 的所有节点
    
    高级用法：
    1. G.out_degree() 返回所有节点的 (节点, 出度) 元组迭代器
    2. 出度为 0 表示该节点不依赖任何其他节点（叶子节点）
    3. 在 DAG 中，出度为 0 的节点是最终输出节点
    
    Parameters
    ----------
    G : nx.DiGraph
        有向图
    
    Returns
    -------
    list
        出度为 0 的节点列表
    """
    return [v for v, d in G.out_degree() if d == 0]


def skip_node(G: nx.DiGraph, node):
    """跳过中间节点，将两端的节点直接连接起来
    
    高级用法：
    1. G.pred[node] 获取节点的前驱节点字典（入边来源）
    2. G.succ[node] 获取节点的后继节点字典（出边目标）
    3. product(pred, succ) 生成笛卡尔积，连接所有前驱和后继
    4. 这是图的简化操作，用于消除中间节点
    
    示例：
    1. (A,B,C) 模式，直接成 (A,C)
    2. (A,B,C), (D, B) 模式，变成 (A,C),(D,C)
    
    Parameters
    ----------
    G : nx.DiGraph
        有向图
    node : hashable
        要跳过的节点
    
    Returns
    -------
    nx.DiGraph
        修改后的图
    
    注意：
    使用 product 生成多个关联边，这是 itertools 的高级用法
    """
    pred = G.pred[node]
    succ = G.succ[node]
    if len(pred) == 0 or len(succ) == 0:
        return G
    # 这里用了product生成多个关联边
    # product(pred, succ) 生成笛卡尔积，例如 pred=[A,D], succ=[C] -> [(A,C), (D,C)]
    G.add_edges_from(product(pred, succ))
    G.remove_node(node)
    return G


def remove_paths(G: nx.DiGraph, *args):
    """从叶子节点开始删除路径，直到不影响其他分支
    
    高级用法：
    1. 从叶子节点（出度为0）开始反向删除
    2. 使用广度优先搜索（BFS）向上遍历
    3. set(next_pred) 去重，避免重复处理
    4. 多路径删除时，检查节点是否存在（可能已被其他路径删除）
    
    算法说明：
    - 从给定的叶子节点开始
    - 删除出度为0的节点
    - 将其前驱节点加入下一轮处理
    - 重复直到没有可删除的节点
    
    对于Y型结构，会全删除
    
    Parameters
    ----------
    G : nx.DiGraph
        有向图
    *args : hashable
        起始节点列表（通常是叶子节点）
    
    Returns
    -------
    nx.DiGraph
        删除路径后的图
    """
    # 准备一个当前节点列表
    this_pred = args
    # 下一步不为空就继续
    while this_pred:
        next_pred = []
        for node in this_pred:
            # 有可能多路径删除时，已经被删
            if not G.has_node(node):
                continue
            # 出度为0
            if len(G.succ[node]) == 0:
                # 找到所有上游节点
                next_pred.extend(G.pred[node])
                G.remove_node(node)
        # 更新下一次循环
        # set(next_pred) 去重，避免重复处理同一节点
        this_pred = list(set(next_pred))
    return G


def remove_paths_by_zero_outdegree(G: nx.DiGraph, exclude):
    """删除悬空路径（不连接到任何输出节点的路径）
    
    高级用法：
    1. 找到所有出度为0的节点（叶子节点）
    2. 排除指定的节点（通常是最终输出节点）
    3. 删除剩余的悬空路径
    
    应用场景：
    - CSE 后可能产生一些中间变量，但最终没有用到
    - 这些中间变量形成悬空路径，可以安全删除
    - 减少不必要的计算
    
    注意：
    如果没有设置要排除，可能全图被删
    
    Parameters
    ----------
    G : nx.DiGraph
        有向图
    exclude : iterable
        要排除的节点列表（通常是最终输出节点）
    
    Returns
    -------
    nx.DiGraph
        删除悬空路径后的图
    """
    nodes = zero_outdegree(G)
    # 悬空：出度为0但不在排除列表中的节点
    dangling = set(nodes) - set(exclude)
    return remove_paths(G, *dangling)


def show_nodes(G):
    """按拓扑排序的层级显示节点
    
    高级用法：
    1. nx.topological_generations(G) 返回拓扑排序的层级生成器
    2. 每一层包含可以并行执行的节点（无依赖关系）
    3. 用于调试和可视化 DAG 结构
    
    Parameters
    ----------
    G : nx.DiGraph
        有向图
    
    注意：
    图必须是有向无环图（DAG），否则会抛出异常
    """
    for i, generation in enumerate(nx.topological_generations(G)):
        print(i, '=' * 20, generation)
        for node in generation:
            print(G.nodes[node])


def show_paths(G: nx.DiGraph, *args):
    """显示路径
    """
    # 准备一个当前节点列表
    this_pred = args
    # 下一步不为空就继续
    while this_pred:
        next_pred = []
        for node in this_pred:
            print(G.nodes[node])
            next_pred.extend(G.pred[node])

        # 更新下一次循环
        this_pred = list(set(next_pred))
    return G


def node_included_path(G: nx.DiGraph, source):
    """得到节点所在路径的所有节点
    
    高级用法：
    1. nx.ancestors(G, source) 获取节点的所有祖先节点（上游）
    2. nx.descendants(G, source) 获取节点的所有后代节点（下游）
    3. 使用集合的 | 运算符合并集合
    
    应用场景：
    - 查找影响某个节点的所有节点
    - 查找被某个节点影响的所有节点
    - 用于依赖分析和影响范围评估
    
    TODO: 总感觉官方提供了类似方法，就是没找到
    
    Parameters
    ----------
    G : nx.DiGraph
        有向图
    source : hashable
        目标节点
    
    Returns
    -------
    set
        包含该节点及其所有祖先和后代的集合
    """
    pred = nx.ancestors(G, source)
    succ = nx.descendants(G, source)
    # set先后没有区别
    return pred | succ | {source}


# 根据原版做了修改，树结构顶部为根，向下生成。与表达式正好相反，所以这里特意将找节点的方向反过来
# https://stackoverflow.com/questions/29586520/can-one-get-hierarchical-graphs-from-networkx-with-python-3/
def hierarchy_pos(G, root, levels=None, width=1., height=1.):
    """计算层次图的节点位置，用于可视化
    
    高级用法：
    1. 递归算法计算每层的节点数量和位置
    2. 使用 predecessors 而非 successors，因为表达式依赖方向相反
    3. parent 参数防止递归中的循环引用
    4. 动态计算垂直间距（vert_gap）和水平间距（dx）
    
    算法说明：
    - make_levels: 递归计算每层的节点总数
    - make_pos: 递归计算每个节点的 (x, y) 坐标
    - 同一层节点均匀分布在水平方向
    - 不同层按垂直方向排列
    
    If there is a cycle that is reachable from root, then this will see infinite recursion.
    
    Parameters
    ----------
    G : nx.DiGraph
        有向图
    root : hashable
        根节点
    levels : dict, optional
        预定义的层级字典
    width : float
        绘图宽度
    height : float
        绘图高度
    
    Returns
    -------
    dict
        节点到位置坐标的映射 {node: (x, y)}
    
    注意：
    如果图中存在从 root 可达的环，会导致无限递归
    """
    TOTAL = "total"
    CURRENT = "current"

    def make_levels(levels, node=root, currentLevel=0, parent=None):
        """递归计算每层的节点数量
        
        高级用法：
        1. 递归遍历图结构
        2. parent 参数防止重复访问（避免环）
        3. 使用 predecessors 向上遍历（表达式依赖方向）
        """
        if not currentLevel in levels:
            levels[currentLevel] = {TOTAL: 0, CURRENT: 0}
        levels[currentLevel][TOTAL] += 1
        # 使用 predecessors 而非 successors，因为表达式依赖方向相反
        neighbors = G.predecessors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                levels = make_levels(levels, neighbor, currentLevel + 1, node)
        return levels

    def make_pos(pos, node=root, currentLevel=0, parent=None, vert_loc=0):
        """递归计算每个节点的位置坐标
        
        高级用法：
        1. dx = 1 / levels[currentLevel][TOTAL] 计算水平间距
        2. levels[currentLevel][CURRENT] 跟踪当前层的节点索引
        3. vert_loc 从上到下递减，实现垂直布局
        """
        dx = 1 / levels[currentLevel][TOTAL]
        left = dx / 2
        # 计算节点的 x 坐标：基于当前层的位置
        pos[node] = ((left + dx * levels[currentLevel][CURRENT]) * width, vert_loc)
        levels[currentLevel][CURRENT] += 1
        neighbors = G.predecessors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                # vert_loc - vert_gap 向下移动
                pos = make_pos(pos, neighbor, currentLevel + 1, node, vert_loc - vert_gap)
        return pos

    if levels is None:
        # 动态计算层级
        levels = make_levels({})
    else:
        # 使用预定义的层级
        levels = {l: {TOTAL: levels[l], CURRENT: 0} for l in levels}
    # 计算垂直间距
    vert_gap = height / (max([l for l in levels]) + 1)
    return make_pos({})
