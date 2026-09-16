import argparse
import os
from typing import Sequence, Literal

import jinja2
from jinja2 import FileSystemLoader, TemplateNotFound

from expr_codegen.expr import TS, CS, GP
from expr_codegen.model import ListDictList
from expr_codegen.polars.printer import PolarsStrPrinter


def get_groupby_from_tuple(tup, func_name, drop_cols):
    """根据分组类型生成 Polars 分组代码
    
    高级用法：
    1. 根据元组前缀判断分组类型（TS/CS/GP）
    2. 生成对应的 sort 和 over 代码
    3. 处理 drop_cols 删除中间变量
    
    分组类型说明：
    - TS (Time Series): 按资产分组，按时间排序，使用 over(_ASSET_, order_by=_DATE_)
    - CS (Cross Section): 按日期分组，使用 over(_DATE_)
    - GP (Group): 按日期和自定义分组，使用 over(_DATE_, group)
    
    Parameters
    ----------
    tup : tuple
        分组元组，如 ('ts', 'asset') 或 ('cs', 'date')
    func_name : str
        函数名
    drop_cols : list
        要删除的列名列表
    
    Returns
    -------
    str
        生成的 Polars 代码
    """
    prefix2, *_ = tup

    if len(drop_cols) > 0:
        drop_str = f'.drop(*{drop_cols})'
    else:
        drop_str = ""

    if prefix2 == TS:
        # 组内需要按时间进行排序，需要维持顺序
        prefix2, asset = tup
        return f'df = {func_name}(df.sort(_ASSET_, _DATE_)){drop_str}'
    if prefix2 == CS:
        prefix2, date = tup
        return f'df = {func_name}(df.sort(_DATE_)){drop_str}'
    if prefix2 == GP:
        prefix2, date, group = tup
        return f'df = {func_name}(df.sort(_DATE_, "{group}")){drop_str}'

    return f'df = {func_name}(df){drop_str}'


def symbols_to_code(syms):
    """将符号列表转换为 Polars 列引用代码
    
    高级用法：
    1. 生成字符串列表和带引号的字符串列表
    2. 使用 pl.col 创建列引用
    
    转换示例：
    - ['CLOSE', 'OPEN'] -> pl.col('CLOSE'), pl.col('OPEN')
    
    Parameters
    ----------
    syms : list
        符号列表
    
    Returns
    -------
    str
        生成的代码字符串
    """
    a = [f"{s}" for s in syms]
    b = [f"'{s}'" for s in syms]
    return f"""_ = [{','.join(b)}]
[{','.join(a)}] = [pl.col(i) for i in _]"""


def codegen(exprs_ldl: ListDictList, exprs_src, syms_dst,
            filename,
            date='date', asset='asset',
            extra_codes: Sequence[str] = (),
            over_null: Literal['order_by', 'partition_by', None] = 'partition_by',
            ge_date_idx: int = 0,
            **kwargs):
    """基于 Jinja2 模板的 Polars 代码生成
    
    高级用法：
    1. 使用 PolarsStrPrinter 打印表达式
    2. 根据 ListDictList 生成分层函数
    3. 使用 argparse 解析注释中的参数
    4. 处理 over_null 参数控制窗口函数行为
    
    over_null 参数说明：
    - 'partition_by': 将空值检查放在 PARTITION BY 中
    - 'order_by': 将空值检查放在 ORDER BY 中
    - None: 不处理空值
    
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
    ge_date_idx : int
        过滤最后 N 天的索引
    
    Returns
    -------
    str
        生成的代码字符串
    """
    if filename is None:
        filename = 'template.py.j2'

    parser = argparse.ArgumentParser()
    parser.add_argument("--over_null", type=str, nargs="?", default=over_null)

    # 打印Polars风格代码
    p = PolarsStrPrinter()

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
    for i, row in enumerate(exprs_ldl.values()):
        for k, vv in row.items():
            j += 1
            if len(vv) == 0:
                continue
            # 函数名
            func_name = f'func_{i}_{"__".join(k)}'
            func_code = []
            for kv in vv:
                if kv is None:
                    func_code.append(f"    )")
                    func_code.append(f"# " + '=' * 40)
                    func_code.append(f"    df = df.with_columns(")
                    exprs_dst.append(f"#" + '=' * 40 + func_name)
                else:
                    va, ex, sym, comment = kv
                    # 多个#时，只取第一个#后的参数
                    args, argv = parser.parse_known_args(args=comment.split("#")[1].split(" "))
                    s1 = str(ex)
                    s2 = p.doprint(ex)
                    if s1 != s2:
                        # 不想等，打印注释，显示会更直观察
                        func_code.append(f"# {va} = {s1}")
                    if k[0] == TS:
                        ts_func_name = func_name
                        # https://github.com/pola-rs/polars/issues/12925#issuecomment-2552764629
                        _sym = [f"{s}.is_not_null()" for s in set(sym)]
                        if len(_sym) > 1:
                            _sym = f"pl.all_horizontal({','.join(_sym)})"
                        else:
                            _sym = ','.join(_sym)

                        if len(_sym) == 0:
                            func_code.append(f"{va}=({s2}).over(_ASSET_, order_by=_DATE_),")
                        else:
                            if args.over_null == 'partition_by':
                                func_code.append(f"{va}=({s2}).over({_sym}, _ASSET_, order_by=_DATE_),")
                            elif args.over_null == 'order_by':
                                func_code.append(f"{va}=({s2}).over(_ASSET_, order_by=[{_sym}, _DATE_]),")
                            else:
                                func_code.append(f"{va}=({s2}).over(_ASSET_, order_by=_DATE_),")

                    elif k[0] == CS:
                        func_code.append(f"{va}=({s2}).over(_DATE_),")
                    elif k[0] == GP:
                        func_code.append(f"{va}=({s2}).over(_DATE_, '{k[2]}'),")
                    else:
                        func_code.append(f"{va}={s2},")
                    exprs_dst.append(f"{va} = {s1} {comment}")
                    if va not in syms_dst:
                        syms_out.append(va)
            func_code.append(f"    )")
            func_code = func_code[1:]

            # polars风格代码列表
            funcs[func_name] = '\n'.join(func_code)
            # 只有下划线开头再删除
            ds = [x for x in drop_symbols[j] if x.startswith('_')]
            # 分组应用代码
            groupbys[func_name] = get_groupby_from_tuple(k, func_name, ds)

    syms1 = symbols_to_code(syms_dst)
    syms2 = symbols_to_code(syms_out)
    # filter_last处理
    _groupbys = {'sort': groupbys['sort']}
    if ts_func_name is None:
        _groupbys['_filter_last'] = "df = _filter_last(df, ge_date_idx)"
    for k, v in groupbys.items():
        _groupbys[k] = v
        if k == ts_func_name:
            _groupbys[k + '_filter_last'] = "df = _filter_last(df, ge_date_idx)"
    groupbys = _groupbys

    try:
        env = jinja2.Environment(loader=FileSystemLoader(os.path.dirname(__file__)))
        template = env.get_template(filename)
    except TemplateNotFound:
        env = jinja2.Environment(loader=FileSystemLoader(os.path.dirname(filename)))
        template = env.get_template(os.path.basename(filename))

    return template.render(funcs=funcs, groupbys=groupbys,
                           exprs_src=exprs_src, exprs_dst=exprs_dst,
                           syms1=syms1, syms2=syms2,
                           date=date, asset=asset,
                           extra_codes=extra_codes)
