import inspect
import os
import pathlib
from functools import lru_cache
from io import TextIOBase
from typing import Sequence, Union, TypeVar, Optional, Literal, Iterable, Dict, Tuple

import polars as pl
from black import Mode, format_str
from loguru import logger
from sympy import simplify, cse, symbols, numbered_symbols
from sympy.core.expr import Expr
from sympy.logic import boolalg
from sympy.simplify import cse_opts

from expr_codegen.codes import sources_to_exprs
from expr_codegen.expr import get_current_by_prefix, get_children, replace_exprs
from expr_codegen.model import dag_start, dag_end, dag_middle, _RESERVED_WORD_

try:
    from pandas import DataFrame as _pd_DataFrame
except ImportError:
    _pd_DataFrame = None

try:
    from polars import DataFrame as _pl_DataFrame
    from polars import LazyFrame as _pl_LazyFrame
except ImportError:
    _pl_DataFrame = None
    _pl_LazyFrame = None

# 高级用法：TypeVar 定义类型变量，用于类型注解
# 表示 DataFrame 可以是 _pl_LazyFrame、_pl_DataFrame 或 _pd_DataFrame 中的任意一种
# 这样可以在类型检查时支持多种 DataFrame 类型
DataFrame = TypeVar('DataFrame', _pl_LazyFrame, _pl_DataFrame, _pd_DataFrame)

# ===============================
# TypeError: expecting bool or Boolean, not `ts_delay(X, 3)`.
# ts_delay(X, 3) & ts_delay(Y, 3)
# 高级用法：修改 sympy 内部函数，绕过类型检查
# sympy 的 boolalg.as_Boolean 会检查参数是否为布尔类型，但我们的表达式中可能包含非布尔类型的函数
# 通过替换为 lambda x: x，跳过类型检查，允许任意表达式参与布尔运算
boolalg.as_Boolean = lambda x: x


# AttributeError: 'StrictGreaterThan' object has no attribute 'diff'
# ts_count(open > 1, 2) == 2
# 高级用法：动态为 sympy.Expr 类添加方法
# sympy 的比较表达式（如 StrictGreaterThan）默认没有 diff 方法
# 通过动态添加 _diff 方法，支持对比较表达式求导
# evaluate=False 阻止求导时的自动化简，保持表达式结构
def _diff(self, *symbols, **assumptions):
    assumptions.setdefault("evaluate", False)
    from sympy.core.function import _derivative_dispatch
    return _derivative_dispatch(self, *symbols, **assumptions)


# 动态为 sympy.Expr 类添加 diff 方法
# 这是 Python 的猴子补丁（monkey patching）技术，在运行时修改类的方法
Expr.diff = _diff


# ===============================

def simplify2(expr, skip_simplify: bool):
    """带跳过选项的表达式简化
    
    高级用法：
    1. skip_simplify 参数用于遗传算法场景
    2. 遗传算法中常出现 OPEN/OPEN 这样的表达式，会被简化成 1
    3. 跳过简化可以保持原始表达式结构，便于后续处理
    
    Parameters
    ----------
    expr : sympy.Expr
        输入表达式
    skip_simplify : bool
        是否跳过简化
    
    Returns
    -------
    sympy.Expr
        简化后的表达式（或原表达式）
    """
    # OPEN/OPEN会被简化成1，遗传算法中常出现，可以跳过简化
    if skip_simplify:
        return expr
    try:
        expr = simplify(expr)
    except (AttributeError, TypeError) as e:
        logger.warning('{}, 表达式无法简化, {}', expr, repr(e))
    return expr


class ExprTool:

    def __init__(self):
        self.get_current_func = get_current_by_prefix
        self.get_current_func_kwargs = {}
        self.exprs_list = {}
        self.exprs_names = []
        self.globals_ = {}

    def set_current(self, func, **kwargs):
        self.get_current_func = func
        self.get_current_func_kwargs = kwargs

    def extract(self, expr, date, asset):
        """抽取分割后的子公式

        Parameters
        ----------
        expr
            单表达式

        Returns
        -------
        表达式列表

        """
        exprs = []
        syms = []
        get_children(self.get_current_func, self.get_current_func_kwargs,
                     expr,
                     output_exprs=exprs, output_symbols=syms,
                     date=date, asset=asset)
        # print('=' * 20, expr)
        # print(exprs)
        return exprs, syms

    def merge(self, date, asset, args, skip_simplify):
        """合并多个表达式

        1. 先抽取分割子公式
        2. 合并 子公式+长公式，去重

        Parameters
        ----------
        args
            表达式列表

        Returns
        -------
        表达式列表
        """
        # 抽取前先化简
        args = [(k, simplify2(v, skip_simplify), c) for k, v, c in args]

        # 保留了注释信息
        exprs_syms = [(self.extract(v, date, asset), c) for k, v, c in args]
        exprs = []
        syms = []
        for (e, s), c in exprs_syms:
            syms.extend(s)
            for _ in e:
                # 抽取的表达式添加注释
                exprs.append((_, c))

        syms = sorted(set(syms), key=syms.index)
        # 如果目标有重复表达式，这里会混乱
        exprs = sorted(set(exprs), key=exprs.index)
        # 这里不能合并简化与未简化的表达式，会导致cse时失败，需要简化表达式合并
        exprs = exprs + [(v, c) for k, v, c in args]

        # print(exprs)
        syms = [str(s) for s in syms]
        return exprs, syms

    def reduce(self, repl, redu):
        """减少中间变量数量，有利用减少内存占用"""

        exprs_list = []

        # cse前简化一次，cse后不再简化
        # (~开盘涨停 & 昨收涨停) | (~收盘涨停 & 最高涨停)
        for k, v in repl:
            exprs_list.append((k, v, "#"))
        for k, v, c in redu:
            exprs_list.append((k, v, c))

        return exprs_list

    def cse(self, exprs, symbols_repl=None, exprs_src=None, skip_simplify=False):
        """公共子表达式消除（Common Subexpression Elimination）
        
        高级用法：
        1. 使用 sympy.cse 提取重复的表达式，减少重复计算
        2. symbols_repl=numbered_symbols('_x_') 生成带编号的中间变量
        3. optimizations 参数控制优化策略
        4. repl 是替换列表（公共表达式），redu 是简化后的表达式
        
        核心算法：
        - cse 会将重复的表达式提取出来，用变量替换
        - 例如：ts_sum(x, 10) + ts_sum(x, 10) 会变成 _x_0 + _x_0
        - 其中 _x_0 = ts_sum(x, 10)
        
        Parameters
        ----------
        exprs : list
            表达式列表
        symbols_repl : iterator
            中间字段名迭代器，如 numbered_symbols('_x_')
        exprs_src : list
            最终字段名列表
        skip_simplify : bool
            是否跳过优化
        
        Returns
        -------
        list
            合并后的表达式列表（包含公共表达式和最终表达式）
        
        注意：
        对于表达式右边相同，左边不同的情况，会当成一个处理
        例如：A = x + y, B = x + y，会被识别为同一个公共表达式
        """
        self.exprs_names = [k for k, v, c in exprs_src]
        # 包含了注释信息
        _exprs = [k for k, v in exprs]

        # 注意：对于表达式右边相同，左边不同的情况，会当成一个处理
        if skip_simplify:
            # 跳过优化，直接进行 CSE
            repl, redu = cse(_exprs, symbols_repl, optimizations=[])
        else:
            # 使用 sub_pre 和 sub_post 优化策略
            # sub_pre: 替换前的优化
            # sub_post: 替换后的优化
            repl, redu = cse(_exprs, symbols_repl, optimizations=[(cse_opts.sub_pre, cse_opts.sub_post), ])

        outputs_len = len(exprs_src)

        new_redu = []
        symbols_redu = iter(exprs_src)
        for expr in redu[-outputs_len:]:
            # 可能部分表达式只在之前出现过，后面完全用不到如，ts_rank(ts_decay_linear(x_147, 11.4157), 6.72611)
            # 只取最后 outputs_len 个表达式（对应最终输出）
            variable = next(symbols_redu)
            a = symbols(variable[0])
            new_redu.append((a, expr, variable[2]))

        self.exprs_list = self.reduce(repl, new_redu)

        # with open("exprs.pickle", "wb") as file:
        #     pickle.dump(exprs_dict, file)

        return self.exprs_list

    def dag(self, merge: bool, skip_columns, date, asset, skip_simplify):
        """生成DAG"""
        G = dag_start(self.exprs_list, self.get_current_func, self.get_current_func_kwargs, date, asset)
        if merge:
            G = dag_middle(G, self.exprs_names, skip_columns, self.get_current_func, self.get_current_func_kwargs, date, asset, skip_simplify)
        return dag_end(G)

    def all(self, exprs_src, style: Literal['pandas', 'polars', 'sql', 'rust'] = 'polars',
            template_file: Optional[str] = None,
            replace: bool = True, regroup: bool = False, format: bool = True,
            date='date', asset='asset',
            extra_codes: Sequence[object] = (),
            over_null: Literal['order_by', 'partition_by', None] = 'partition_by',
            table_name: str = 'self',
            ge_date_idx: int = 0,
            skip_simplify: bool = False,
            skip_columns: Iterable[str] = (),
            **kwargs):
        """功能集成版，将几个功能写到一起方便使用

        Parameters
        ----------
        exprs_src: list
            表达式列表
        style: str
            代码风格。可选值 ('polars', 'pandas', 'sql')
        template_file: str
            根据需求可定制模板
        replace:bool
            表达式提换
        regroup:bool
            分组重排。注意：目前好像不稳定
        format:bool
            代码格式化
        date:str
            日期字段名
        asset:str
            资产字段名
        extra_codes: Sequence[object]
            需要复制到模板中的额外代码
        table_name
        ge_date_idx
        skip_simplify
        skip_columns
            数据中已经存在的列不再参与计算，直接使用历史值

        Returns
        -------
        代码字符串

        """
        assert style in ('pandas', 'polars', 'sql', 'rust')

        if replace:
            exprs_src = replace_exprs(exprs_src)

        # 子表达式在前，原表式在最后
        exprs_dst, syms_dst = self.merge(date, asset, exprs_src, skip_simplify=skip_simplify)
        syms_dst = list(set(syms_dst) - _RESERVED_WORD_)

        # 提取公共表达式
        self.cse(exprs_dst, symbols_repl=numbered_symbols('_x_'), exprs_src=exprs_src, skip_simplify=skip_simplify)
        # 有向无环图流转
        exprs_ldl, G = self.dag(True, skip_columns, date, asset, skip_simplify=skip_simplify)

        if regroup:
            exprs_ldl.optimize(merge=style != 'sql')

        if style == 'polars':
            from expr_codegen.polars.code import codegen
        elif style == 'pandas':
            from expr_codegen.pandas.code import codegen
        elif style == 'sql':
            from expr_codegen.sql.code import codegen
            format = False
        elif style == 'rust':
            from expr_codegen.rust.code import codegen
            format = False
        else:
            raise ValueError(f'unknown style {style}')

        extra_codes = [c if isinstance(c, str) else inspect.getsource(c) for c in extra_codes]

        codes = codegen(exprs_ldl, exprs_src, syms_dst,
                        filename=template_file, date=date, asset=asset,
                        extra_codes=extra_codes,
                        over_null=over_null,
                        table_name=table_name,
                        ge_date_idx=ge_date_idx,
                        **kwargs)

        logger.info(f'{style} code is generated')

        if format:
            # 格式化。在遗传算法中没有必要
            # 高级用法：使用 black 库的 format_str 进行代码格式化
            # line_length=600 设置行长限制（比默认的 88 更大，适合生成的复杂表达式）
            # magic_trailing_comma=True 在多行结构中自动添加尾随逗号
            codes = format_str(codes, mode=Mode(line_length=600, magic_trailing_comma=True))

        return codes, G

    @lru_cache(maxsize=64)
    def _get_code(self,
                  source: str, *more_sources: str,
                  extra_codes: str,
                  output_file: str,
                  convert_xor: bool,
                  style: Literal['pandas', 'polars', 'sql', 'rust'] = 'polars',
                  template_file: Optional[str] = None,
                  date: str = 'date', asset: str = 'asset',
                  over_null: Literal['order_by', 'partition_by', None] = 'partition_by',
                  table_name: str = 'self',
                  ge_date_idx: int = 0,
                  skip_simplify: bool = False,
                  skip_columns: Iterable[str] = (),
                  **kwargs) -> str:
        """通过字符串生成代码，带缓存功能
        
        高级用法：
        1. @lru_cache(maxsize=64) 装饰器缓存结果，相同输入直接返回缓存
        2. 缓存基于所有参数的哈希值，包括字符串和可哈希对象
        3. 适用于遗传算法等场景，避免重复生成相同代码
        
        注意事项：
        - 缓存可能导致调试时看不到最新代码
        - 需要留意日志输出，确认是否使用了缓存
        - maxsize=64 表示最多缓存 64 个不同的输入组合
        
        Parameters
        ----------
        source : str
        主要代码字符串
        *more_sources : str
        额外的代码字符串
        extra_codes : str
            额外代码（如正则表达式、辅助函数）
        output_file : str
            输出文件路径
        convert_xor : bool
            是否将 ^ 转换为乘方
        style : str
            代码风格
        
        Returns
        -------
        str
            生成的代码字符串
        """
        raw, exprs_list = sources_to_exprs(self.globals_, source, *more_sources, convert_xor=convert_xor)

        # 生成代码
        code, G = _TOOL_.all(exprs_list, style=style, template_file=template_file,
                             replace=True, regroup=True, format=True,
                             date=date, asset=asset,
                             # 复制了需要使用的函数，还复制了最原始的表达式
                             extra_codes=(raw,
                                          # 传入多个列的方法
                                          extra_codes,
                                          ),
                             over_null=over_null,
                             table_name=table_name,
                             ge_date_idx=ge_date_idx,
                             skip_simplify=skip_simplify,
                             skip_columns=skip_columns,
                             **kwargs)

        # 移回到cache，防止多次调用多次保存
        if hasattr(output_file, "write"):
            # output_file 是文件对象（如 StringIO）
            output_file.write(code)
        elif output_file is not None:
            # output_file 是文件路径字符串
            output_file = pathlib.Path(output_file)
            logger.info(f'save to {output_file.absolute()}')
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(code)

        return code


@lru_cache(maxsize=64, typed=True)
def _get_func_from_code_py(code: str):
    """从代码字符串动态获取函数
    
    高级用法：
    1. exec(code, globals_) 动态执行 Python 代码字符串
    2. typed=True 表示区分参数类型（如 str 和 bytes 会视为不同缓存）
    3. 假设生成的代码中有一个名为 main 的函数
    
    安全注意事项：
    - exec 可以执行任意代码，仅用于可信的代码生成场景
    - 不要对用户输入的代码使用 exec
    
    Parameters
    ----------
    code : str
        Python 代码字符串
    
    Returns
    -------
    callable
        代码中定义的 main 函数
    """
    logger.info(f'get func from code py')
    globals_ = {}
    # 动态执行代码字符串，在 globals_ 命名空间中定义函数
    exec(code, globals_)
    # 返回 main 函数
    return globals_['main']


@lru_cache(maxsize=64, typed=True)
def _get_func_from_module(module: str):
    """从模块动态导入函数（可下断点调试）
    
    高级用法：
    1. __import__(module, fromlist=['*']) 动态导入模块
    2. fromlist=['*'] 确保导入模块本身而非包
    3. 适用于需要调试的场景，可以直接在模块文件中设置断点
    
    优势：
    - 比 exec 更安全，因为是从文件导入
    - 支持 IDE 断点调试
    - 代码可读性更好
    
    Parameters
    ----------
    module : str
        模块名（如 'my_module' 或 'package.module'）
    
    Returns
    -------
    callable
        模块中定义的 main 函数
    
    注意：
    可能调用到其他目录下的同名模块，所以保存的文件名要有辨识度
    """
    m = __import__(module, fromlist=['*'])
    logger.info(f'get func from module {m}')
    return m.main


@lru_cache(maxsize=64, typed=True)
def _get_func_from_file_py(file: str):
    file = pathlib.Path(file)
    logger.info(f'get func from file "{file.absolute()}"')
    with open(file, 'r', encoding='utf-8') as f:
        globals_ = {}
        exec(f.read(), globals_)
        return globals_['main']


@lru_cache(maxsize=64, typed=True)
def _get_code_from_file(file: str):
    file = pathlib.Path(file)
    logger.info(f'get code from file "{file.absolute()}"')
    with open(file, 'r', encoding='utf-8') as f:
        return f.read()


_TOOL_ = ExprTool()


def codegen_exec(df: Union[DataFrame, None],
                 *codes,
                 over_null: Literal['partition_by', 'order_by', None],
                 extra_codes: str = r'CS_SW_L1 = r"^sw_l1_\d+$"',
                 output_file: Union[str, TextIOBase, None] = None,
                 run_file: Union[bool, str] = False,
                 convert_xor: bool = False,
                 style: Literal['pandas', 'polars', 'sql', 'rust'] = 'polars',
                 template_file: Optional[str] = None,
                 date: str = 'date', asset: str = 'asset',
                 table_name: str = 'self',
                 ge_date_idx: int = 0,
                 skip_simplify: bool = False,
                 skip_columns: Iterable[str] = (),
                 **kwargs) -> Union[DataFrame, str]:
    r"""快速转换源代码并执行

    Parameters
    ----------
    df: pl.DataFrame, pd.DataFrame, pl.LazyFrame,None
        输入DataFrame，输出DataFrame
        输入None，输出代码
    codes:
        函数体。此部分中的表达式会被翻译成目标代码
    extra_codes: str
        额外代码。不做处理，会被直接复制到目标代码中。例如：

        r'CS_SW_L1 = r"^sw_l1_\d+$"'

        apply_const_to_expr()

    output_file: str| TextIOBase
        保存生成的目标代码到文件中
    run_file: bool or str
        是否不生成脚本，直接运行代码。注意：带缓存功能，多次调用不重复生成
        - 如果是True，会自动从output_file中读取代码
        - 如果是字符串，会自动从run_file中读取代码
        - 如果是模块名，会自动从模块中读取代码(可调试)
            - 注意：可能调用到其他目录下的同名模块，所以保存的文件名要有辨识度
        - 如果文件不存在，先生成文件。第二次从生成的文件中生成
    convert_xor: bool
        ^ 转成异或还是乘方
    style: str
        代码风格。可选值 'pandas', 'polars', 'sql'
        - pandas: 不支持struct
        - sql: 只生成sql语句，不执行
    template_file: str
        代码模板
    date: str
        时间字段
    asset: str
        资产字段
    over_null: str
        时序中遇到null时的处理方式。只在style参数为'polars', 'sql'时有效
        - partition_by: 空值划分到不同分区
        - order_by: 空值排同一分区的前排
        - None: 不做处理
    table_name:str
        表名。只在style参数为sql时有效
    ge_date_idx:int
        在实盘时，只需要最后一两天日期的数据，可以在最后一个`ts`之后过滤数据。目前只在style参数为'polars', 'pandas'时有效
        0 表示不过滤 >=date[0]
        -1 表示最新一天 >=date[-1]
        -2 表示最近两天 >=date[-2]
    skip_simplify:bool
        遗传算法时很有可能出现OPEN/OPEN，可以跳过化简步骤
        1. 跳过cse前的simplify
        2. 跳过cse时的optimizations
        3. 跳过DAG中的部分merge步骤
    skip_columns:
        已经存在的列不参与计算。可用于加快计算速度。只在计算耗时久时再用，否则没有必要
        例如：在研发阶段，第一次计算100个因子，第二次，只改动了其中的5个，所以只要将这5个从df.columns中排除即可。
        注意：生成的源代码有差异。

    Returns
    -------
    DataFrame
        输出DataFrame
    str
        输出代码

    Notes
    -----
    处处都有缓存，所以在公式研发阶段要留意日志输出。以免一直调试的旧公式

    1. 确保重新生成了代码  `code is generated`
    2. 通过代码得到了函数 `get func from code`

    """
    skip_columns = tuple(skip_columns)

    if df is not None:
        input_file = None
        # 以下代码都带缓存功能
        if run_file is True:
            assert output_file is not None, 'output_file is required'
            input_file = str(output_file)
        elif run_file is not False:
            input_file = str(run_file)

        if input_file is not None:
            if input_file.endswith('.py'):
                if os.path.exists(input_file):
                    return _get_func_from_file_py(input_file)(df, ge_date_idx)
            elif input_file.endswith('.sql'):
                if os.path.exists(input_file):
                    with pl.SQLContext(frames={table_name: df}) as ctx:
                        return ctx.execute(_get_code_from_file(input_file), eager=isinstance(df, _pl_DataFrame))
            else:
                return _get_func_from_module(input_file)(df, ge_date_idx)  # 可断点调试
    else:
        pass

    # 此代码来自于sympy.var
    # 高级用法：使用 inspect.currentframe().f_back 获取调用者的全局命名空间
    # 这样可以访问调用者代码中定义的符号（如 OPEN, CLOSE 等）
    # f_back 表示调用者的栈帧，f_globals 是该栈帧的全局命名空间
    frame = inspect.currentframe().f_back
    _TOOL_.globals_ = frame.f_globals.copy()
    # 删除 frame 引用，避免循环引用导致的内存泄漏
    del frame

    more_sources = [c if isinstance(c, str) else inspect.getsource(c) for c in codes]
    # 高级用法：inspect.getsource(c) 获取函数或类的源代码字符串
    # 这样可以传入函数对象，自动提取其源代码进行转换
    # 支持混合传入字符串和函数对象

    code = _TOOL_._get_code(
        *more_sources,
        extra_codes=extra_codes,
        output_file=output_file,
        convert_xor=convert_xor,
        style=style, template_file=template_file,
        date=date, asset=asset,
        over_null=over_null,
        table_name=table_name,
        ge_date_idx=ge_date_idx,
        skip_simplify=skip_simplify,
        skip_columns=skip_columns,
        **kwargs
    )

    if df is None:
        # 如果df为空，直接返回代码
        return code
    elif style == 'rust':
        return code
    elif style == 'sql':
        with pl.SQLContext(frames={table_name: df}) as ctx:
            return ctx.execute(code, eager=isinstance(df, _pl_DataFrame))
    else:
        # 代码一样时就从缓存中取出函数
        return _get_func_from_code_py(code)(df, ge_date_idx)
