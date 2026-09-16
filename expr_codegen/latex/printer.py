from sympy import Symbol, Function, Basic
from sympy.core.sorting import default_sort_key
from sympy.printing.latex import LatexPrinter, accepted_latex_functions


class ExprLatexPrinter(LatexPrinter):
    """LaTeX 表达式打印机，解决下划线被转成下标的问题
    
    高级用法：
    1. 继承 sympy.LatexPrinter
    2. 重写 _hprint_Function 处理函数名中的下划线
    3. 重写 _print_Symbol 处理符号中的下划线
    
    主要解决的问题：
    - 默认的 LatexPrinter 会将下划线 _ 转换为下标
    - 这会导致函数名如 ts_mean 被错误渲染
    - 本打印机将 _ 替换为 \_ 保持原样
    
    注意：
    - 用于生成数学公式显示
    - 在 VSCode 或 Notebook 中可正常显示
    """

    def _print(self, expr, **kwargs) -> str:
        """Internal dispatcher

        Tries the following concepts to print an expression:
            1. Let the object print itself if it knows how.
            2. Take the best fitting method defined in the printer.
            3. As fall-back use the emptyPrinter method for the printer.
        """
        self._print_level += 1
        try:
            # If the printer defines a name for a printing method
            # (Printer.printmethod) and the object knows for itself how it
            # should be printed, use that method.
            if self.printmethod and hasattr(expr, self.printmethod):
                if not (isinstance(expr, type) and issubclass(expr, Basic)):
                    return getattr(expr, self.printmethod)(self, **kwargs)

            # See if the class of expr is known, or if one of its super
            # classes is known, and use that print function
            # Exception: ignore the subclasses of Undefined, so that, e.g.,
            # Function('gamma') does not get dispatched to _print_gamma
            classes = type(expr).__mro__
            # if AppliedUndef in classes:
            #     classes = classes[classes.index(AppliedUndef):]
            # if UndefinedFunction in classes:
            #     classes = classes[classes.index(UndefinedFunction):]
            # Another exception: if someone subclasses a known function, e.g.,
            # gamma, and changes the name, then ignore _print_gamma
            if Function in classes:
                i = classes.index(Function)
                classes = tuple(c for c in classes[:i] if \
                                c.__name__ == classes[0].__name__ or \
                                c.__name__.endswith("Base")) + classes[i:]
            for cls in classes:
                printmethodname = '_print_' + cls.__name__
                printmethod = getattr(self, printmethodname, None)
                if printmethod is not None:
                    return printmethod(expr, **kwargs)
            # Unknown object, fall back to the emptyPrinter.
            return self.emptyPrinter(expr)
        finally:
            self._print_level -= 1

    def _hprint_Function(self, func: str) -> str:
        """打印函数名，处理下划线
        
        高级用法：
        1. 将下划线替换为 \_ 避免被转成下标
        2. 检查是否为已接受的 LaTeX 函数
        3. 使用 \operatorname 包装非标准函数名
        
        转换示例：
        - ts_mean -> \operatorname{ts_mean}
        - abs -> \abs (已接受)
        """
        # func = self._deal_with_super_sub(func)
        superscriptidx = -1  # func.find("^")
        subscriptidx = -1  # func.find("_")
        func = func.replace('_', r'\_')
        if func in accepted_latex_functions:
            name = r"\%s" % func
        elif len(func) == 1 or func.startswith('\\') or subscriptidx == 1 or superscriptidx == 1:
            name = func
        else:
            if superscriptidx > 0 and subscriptidx > 0:
                name = r"\operatorname{%s}%s" % (
                    func[:min(subscriptidx, superscriptidx)],
                    func[min(subscriptidx, superscriptidx):])
            elif superscriptidx > 0:
                name = r"\operatorname{%s}%s" % (
                    func[:superscriptidx],
                    func[superscriptidx:])
            elif subscriptidx > 0:
                name = r"\operatorname{%s}%s" % (
                    func[:subscriptidx],
                    func[subscriptidx:])
            else:
                name = r"\operatorname{%s}" % func
        return name

    def _print_Symbol(self, expr: Symbol, style='plain'):
        """打印符号变量，处理下划线
        
        高级用法：
        1. 检查 symbol_names 设置中是否有自定义名称
        2. 将下划线替换为 \_ 避免被转成下标
        
        转换示例：
        - _x_0 -> \_x\_0
        """
        name: str = self._settings['symbol_names'].get(expr)
        if name is not None:
            return name

        return expr.name.replace('_', r'\_')

    def _print_abs_(self, expr, exp=None):
        return self._print_Abs(expr, exp)

    def _print_log_(self, expr, exp=None):
        return self._print_log(expr, exp)

    def _print_max_(self, expr, exp=None):
        # return self._print_Max(expr, exp)
        args = sorted(expr.args, key=default_sort_key)
        texargs = [r"%s" % self._print(symbol) for symbol in args]
        tex = r"\%s\left(%s\right)" % ('max', ", ".join(texargs))
        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_min_(self, expr, exp=None):
        args = sorted(expr.args, key=default_sort_key)
        texargs = [r"%s" % self._print(symbol) for symbol in args]
        tex = r"\%s\left(%s\right)" % ('min', ", ".join(texargs))
        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex


def latex(expr, mode='equation*', mul_symbol='times', **settings):
    """将表达式转换为 LaTeX 字符串
    
    高级用法：
    1. 使用 ExprLatexPrinter 进行打印
    2. 支持多种 LaTeX 模式
    3. 支持乘法符号自定义
    
    Parameters
    ----------
    expr : sympy.Expr
        要转换的表达式
    mode : str
        LaTeX 模式，如 'equation*', 'inline'
    mul_symbol : str
        乘法符号，如 'times', 'cdot'
    **settings : dict
        其他打印设置
    
    Returns
    -------
    str
        LaTeX 字符串
    """
    settings.update({'mode': mode, 'mul_symbol': mul_symbol})
    return ExprLatexPrinter(settings).doprint(expr)


def display_latex(expr):
    """在 VSCode 或 Notebook 中显示 LaTeX 表达式
    
    高级用法：
    1. 使用 IPython.display 显示 Markdown
    2. 自动转换为 LaTeX 格式
    3. 支持 Jupyter Notebook 和 VSCode
    
    注意：
    - 需要 IPython 环境
    - 在纯 Python 环境中可能无法使用
    """
    from IPython.display import Markdown, display

    return display(Markdown(latex(expr)))
