import argparse
import os
from typing import Sequence, Literal

import jinja2
from jinja2 import FileSystemLoader, TemplateNotFound

from expr_codegen.expr import TS, CS, GP
from expr_codegen.model import ListDictList
from expr_codegen.sql.printer import SQLStrPrinter


def codegen(exprs_ldl: ListDictList, exprs_src, syms_dst,
            filename,
            date='date', asset='asset',
            extra_codes: Sequence[str] = (),
            over_null: Literal['order_by', 'partition_by', None] = 'partition_by',
            table_name: str = 'self',
            ge_date_idx: int = 0,
            **kwargs):
    """基于 Jinja2 模板的 SQL 代码生成
    
    与 polars 的差异：
    - 使用 SQLStrPrinter 打印表达式
    - 生成 SQL CTE（Common Table Expression）结构
    - 使用 OVER(PARTITION BY ... ORDER BY ...) 窗口函数
    - 函数名使用 cte_ 前缀而不是 func_
    - 使用反引号包裹列名
    - 没有 symbols_to_code
    - 使用 template.sql.j2
    - 使用 last_func_name 链接 CTE
    
    SQL 特有语法：
    - WITH ... AS (SELECT ...) 定义 CTE
    - OVER(PARTITION BY ... ORDER BY ...) 窗口函数
    - `x` IS NOT NULL 检查空值
    - AND 连接多个条件
    
    Parameters
    ----------
    exprs_ldl : ListDictList
        分层的表达式列表
    exprs_src : list
        源表达式列表
    syms_dst : list
        目标符号列表
    filename : str
        模板文件名
    date : str
        日期字段名
    asset : str
        资产字段名
    extra_codes : Sequence[str]
        额外的代码片段
    over_null : Literal
        空值处理方式
    table_name : str
        基础表名
    ge_date_idx : int
        过滤最后 N 天的索引（未实现）
    
    Returns
    -------
    str
        生成的 SQL 代码字符串
    """
    if filename is None:
        filename = 'template.sql.j2'

    parser = argparse.ArgumentParser()
    parser.add_argument("--over_null", type=str, nargs="?", default=over_null)

    # 打印Polars风格代码
    p = SQLStrPrinter()

    # polars风格代码
    funcs = {}
    # 分组应用代码。这里利用了字典按插入顺序排序的特点，将排序放在最前
    groupbys = {'sort': ''}
    # 处理过后的表达式
    exprs_dst = []
    syms_out = []
    ts_func_name = None
    drop_symbols = exprs_ldl.drop_symbols()
    j = -1
    last_func_name = table_name
    for i, row in enumerate(exprs_ldl.values()):
        for k, vv in row.items():
            j += 1
            if len(vv) == 0:
                continue
            # 函数名
            func_name = f'cte_{i}_{"__".join(k)}'
            func_code = []
            for kv in vv:
                if kv is None:
                    func_code.append(f"{func_name} AS (SELECT *,")
                    exprs_dst.append(f"#" + '=' * 40 + func_name)
                else:
                    va, ex, sym, comment = kv
                    # 多个#时，只取第一个#后的参数
                    args, argv = parser.parse_known_args(args=comment.split("#")[1].split(" "))
                    s1 = str(ex)
                    s2 = p.doprint(ex)
                    if k[0] == TS:
                        ts_func_name = func_name
                        # https://github.com/pola-rs/polars/issues/12925#issuecomment-2552764629
                        _sym = [f"`{s}` IS NOT NULL" for s in set(sym)]
                        if len(_sym) > 1:
                            _sym = f"({' AND '.join(_sym)})"
                        else:
                            _sym = ','.join(_sym)

                        if len(_sym) == 0:
                            func_code.append(f"{s2} OVER(PARTITION BY `{asset}` ORDER BY `{date}`) AS {va},")
                        else:
                            if args.over_null == 'partition_by':
                                func_code.append(f"{s2} OVER(PARTITION BY {_sym},`{asset}` ORDER BY `{date}`) AS {va},")
                            elif args.over_null == 'order_by':
                                func_code.append(f"{s2} OVER(PARTITION BY `{asset}` ORDER BY {_sym},`{date}`) AS {va},")
                            else:
                                func_code.append(f"{s2} OVER(PARTITION BY `{asset}` ORDER BY `{date}`) AS {va},")

                    elif k[0] == CS:
                        func_code.append(f"{s2} OVER(PARTITION BY `{date}`) AS {va},")
                    elif k[0] == GP:
                        func_code.append(f"{s2} OVER(PARTITION BY `{date}`,`{k[2]}`) AS {va},")
                    else:
                        func_code.append(f"{s2} AS {va},")
                    exprs_dst.append(f"{va} = {s1} {comment}")
                    if va not in syms_dst:
                        syms_out.append(va)
            func_code.append(f"FROM {last_func_name}),")
            last_func_name = func_name

            # sql风格代码列表
            funcs[func_name] = '\n  '.join(func_code)
            # 只有下划线开头再删除
            ds = [x for x in drop_symbols[j] if x.startswith('_')]

    if ge_date_idx:
        # TODO 没有实现
        pass

    try:
        env = jinja2.Environment(loader=FileSystemLoader(os.path.dirname(__file__)))
        template = env.get_template(filename)
    except TemplateNotFound:
        env = jinja2.Environment(loader=FileSystemLoader(os.path.dirname(filename)))
        template = env.get_template(os.path.basename(filename))

    return template.render(funcs=funcs,
                           exprs_src=exprs_src, exprs_dst=exprs_dst,
                           date=date, asset=asset,
                           extra_codes=extra_codes,
                           last_func_name=last_func_name)
