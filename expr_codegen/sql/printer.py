from sympy import Basic, Function, StrPrinter
from sympy.printing.precedence import precedence, PRECEDENCE


# TODO: 如有新添加函数，但表达式有变更才需要在此补充对应的打印代码，否则可以省略

class SQLStrPrinter(StrPrinter):
    """SQL 表达式字符串打印机
    
    高级用法：
    1. 继承 sympy.StrPrinter 实现自定义打印
    2. 使用 convert_funcs 映射函数名到 SQL 函数
    3. 逻辑运算使用 SQL 标准关键字（大写）
    4. 比较符使用 SQL 标准（=, <>）
    
    与 polars 的差异：
    - 函数名映射到 SQL 函数（如 abs_ -> ABS）
    - 逻辑运算使用大写（OR, XOR, AND, NOT）
    - 比较符使用 SQL 标准（=, <>）
    - Symbol 使用反引号包裹
    
    注意：
    - SQL 函数名映射参考 Polars SQL 文档
    """
    # https://docs.pola.rs/api/python/stable/reference/sql/functions/index.html
    # https://github.com/pola-rs/polars/blob/main/crates/polars-sql/src/functions.rs
    # 将polars_ta中的函数转换成SQL中的函数
    convert_funcs = {
        # Math functions
        'abs_': 'ABS',
        'ceiling': 'CEIL',
        'div': 'DIV',
        'exp': 'EXP',
        'floor': 'FLOOR',
        'log': 'LN',
        'log10': 'LOG10',
        'log1p': 'LOG1P',
        'log2': 'LOG2',
        'mod': 'MOD',
        'sign': 'SIGN',
        'sqrt': 'SQRT',
        'power': 'POW',
        'round_': 'ROUND',
        # Trig functions
        'arc_cos': 'ACOS',
        'arc_sin': 'ASIN',
        'arc_tan': 'ATAN',
        'arc_tan2': 'ATAN2',
        'cot': 'COT',
        'cos': 'COS',
        'degrees': 'DEGREES',
        'radians': 'RADIANS',
        'sin': 'SIN',
        'tan': 'TAN',
    }

    def _print(self, expr, **kwargs) -> str:
        """内部分发器：根据表达式类型选择合适的打印方法
        
        与 polars 的差异：
        - 在 _print 方法中添加了函数名映射逻辑
        - 检查 cls.__name__ 是否在 convert_funcs 中
        - 如果在，使用 _print_Rename_ 进行函数名转换
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

                # 所有以gp_开头的函数都转换成cs_开头
                if printmethodname.startswith('_print_gp_'):
                    printmethodname = "_print_gp_"

                # polars_ta中的函数转换成SQL函数
                if cls.__name__ in self.convert_funcs:
                    return self._print_Rename_(expr, self.convert_funcs[cls.__name__])

                printmethod = getattr(self, printmethodname, None)
                if printmethod is not None:
                    return printmethod(expr, **kwargs)
            # Unknown object, fall back to the emptyPrinter.
            return self.emptyPrinter(expr)
        finally:
            self._print_level -= 1

    def _print_Symbol(self, expr):
        """打印符号变量
        
        与 polars 的差异：
        - 使用反引号包裹列名
        - 这是 SQL 标准的列名引用方式
        """
        return f'`{expr.name}`'

    def _print_Equality(self, expr):
        """打印等于表达式
        
        与 polars 的差异：
        - 使用 = 而不是 ==
        - 这是 SQL 标准的比较符
        """
        PREC = precedence(expr)
        return "%s = %s" % (self.parenthesize(expr.args[0], PREC), self.parenthesize(expr.args[1], PREC))

    def _print_Unequality(self, expr):
        """打印不等于表达式
        
        与 polars 的差异：
        - 使用 <> 而不是 !=
        - 这是 SQL 标准的比较符
        """
        PREC = precedence(expr)
        return "%s <> %s" % (self.parenthesize(expr.args[0], PREC), self.parenthesize(expr.args[1], PREC))

    def _print_Or(self, expr):
        """打印或运算表达式
        
        与 polars 的差异：
        - 使用 OR 而不是 |
        - 这是 SQL 标准的逻辑运算符
        """
        PREC = PRECEDENCE["Mul"]
        return " OR ".join(self.parenthesize(arg, PREC) for arg in expr.args)

    def _print_Xor(self, expr):
        """打印异或运算表达式
        
        与 polars 的差异：
        - 使用 XOR 而不是 ^
        - 这是 SQL 标准的逻辑运算符
        """
        PREC = PRECEDENCE["Mul"]
        return " XOR ".join(self.parenthesize(arg, PREC) for arg in expr.args)

    def _print_And(self, expr):
        """打印与运算表达式
        
        与 polars 的差异：
        - 使用 AND 而不是 &
        - 这是 SQL 标准的逻辑运算符
        """
        PREC = PRECEDENCE["Mul"]
        return " AND ".join(self.parenthesize(arg, PREC) for arg in expr.args)

    def _print_Not(self, expr):
        """打印非运算表达式
        
        与 polars 的差异：
        - 使用 NOT 而不是 ~
        - 这是 SQL 标准的逻辑运算符
        """
        PREC = PRECEDENCE["Mul"]
        return "NOT %s" % self.parenthesize(expr.args[0], PREC)

    def _print_gp_(self, expr):
        """gp_函数都转换成cs_函数，但要丢弃第一个参数"""
        new_args = [self._print(arg) for arg in expr.args[1:]]
        func_name = expr.func.__name__[3:]
        return "cs_%s(%s)" % (func_name, ",".join(new_args))

    def _print_Rename_(self, expr, new_name):
        """重命名函数
        
        高级用法：
        1. 将函数名替换为 SQL 函数名
        2. 保留参数列表
        
        转换示例：
        - abs_(x) -> ABS(x)
        - log(x) -> LN(x)
        
        注意：
        - 用于将 polars_ta 函数映射到 SQL 函数
        """
        l = [self._print(o) for o in expr.args]
        return new_name + "(%s)" % ", ".join(l)
