from sympy import Basic, Function, StrPrinter
from sympy.printing.precedence import precedence, PRECEDENCE


# TODO: 如有新添加函数，但表达式有变更才需要在此补充对应的打印代码，否则可以省略

class PolarsStrPrinter(StrPrinter):
    """Polars 表达式字符串打印机
    
    高级用法：
    1. 继承 sympy.StrPrinter 实现自定义打印
    2. 重写 _print 方法实现方法分发逻辑
    3. 使用 MRO（方法解析顺序）查找合适的打印方法
    4. gp_ 函数统一转换为 cs_ 函数
    
    与其他打印器的差异：
    - pandas: Symbol 使用 g[expr.name]，保留字特殊处理
    - rust: Symbol 使用 col()，逻辑运算使用链式调用，数字使用 lit() 包装
    - sql: 函数名映射到 SQL 函数，逻辑运算使用大写，比较符不同
    
    注意：
    - gp_ 函数的第一个参数是分组键，在 polars 中不需要显式传递
    - 因此转换为 cs_ 函数时丢弃第一个参数
    """
    def _print(self, expr, **kwargs) -> str:
        """内部分发器：根据表达式类型选择合适的打印方法
        
        高级用法：
        1. 使用 type(expr).__mro__ 获取方法解析顺序
        2. 沿着 MRO 查找匹配的 _print_ 方法
        3. gp_ 开头的函数统一使用 _print_gp_ 处理
        4. 使用 try-finally 确保 _print_level 正确递减
        
        分发顺序：
        1. 对象自身的打印方法（如果有）
        2. 打印器中定义的 _print_ 方法
        3. 默认的 emptyPrinter
        
        注意：
        - _print_level 用于检测循环引用
        - MRO 过滤掉不相关的基类
        """
        self._print_level += 1
        try:
            # If the printer defines a name for a printing method
            # (Printer.printmethod) and the object knows for itself how it
            # should be printed, use that method.
            # 如果打印机定义了打印方法名称（Printer.printmethod）
            # 且对象知道如何打印自己，则使用该方法
            if self.printmethod and hasattr(expr, self.printmethod):
                if not (isinstance(expr, type) and issubclass(expr, Basic)):
                    return getattr(expr, self.printmethod)(self, **kwargs)

            # See if the class of expr is known, or if one of its super
            # classes is known, and use that print function
            # Exception: ignore the subclasses of Undefined, so that, e.g.,
            # Function('gamma') does not get dispatched to _print_gamma
            # 检查 expr 的类是否已知，或者其父类是否已知，并使用相应的打印函数
            # 异常：忽略 Undefined 的子类，例如 Function('gamma') 不会被分发到 _print_gamma
            classes = type(expr).__mro__
            # if AppliedUndef in classes:
            #     classes = classes[classes.index(AppliedUndef):]
            # if UndefinedFunction in classes:
            #     classes = classes[classes.index(UndefinedFunction):]
            # Another exception: if someone subclasses a known function, e.g.,
            # gamma, and changes the name, then ignore _print_gamma
            # 另一个异常：如果有人子类化了一个已知函数，例如 gamma，并更改了名称，则忽略 _print_gamma
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

                printmethod = getattr(self, printmethodname, None)
                if printmethod is not None:
                    return printmethod(expr, **kwargs)
            # Unknown object, fall back to the emptyPrinter.
            # 未知对象，回退到 emptyPrinter
            return self.emptyPrinter(expr)
        finally:
            self._print_level -= 1

    def _print_Symbol(self, expr):
        """打印符号变量
        
        与其他打印器的差异：
        - polars: 直接返回变量名
        - pandas: 使用 g[expr.name]，保留字特殊处理
        - rust: 使用 col("expr.name")
        - sql: 使用 `expr.name`
        """
        return expr.name

    def _print_Equality(self, expr):
        """打印等于表达式
        
        高级用法：
        1. precedence 获取运算符优先级
        2. parenthesize 根据优先级决定是否加括号
        
        与其他打印器的差异：
        - polars/pandas: 使用 ==
        - sql: 使用 =
        """
        PREC = precedence(expr)
        return "%s == %s" % (self.parenthesize(expr.args[0], PREC), self.parenthesize(expr.args[1], PREC))

    def _print_Unequality(self, expr):
        """打印不等于表达式
        
        与其他打印器的差异：
        - polars/pandas: 使用 !=
        - sql: 使用 <>
        """
        PREC = precedence(expr)
        return "%s != %s" % (self.parenthesize(expr.args[0], PREC), self.parenthesize(expr.args[1], PREC))

    def _print_Or(self, expr):
        """打印或运算表达式
        
        与其他打印器的差异：
        - polars/pandas: 使用 |
        - rust: 使用链式 .or()
        - sql: 使用 OR
        """
        PREC = PRECEDENCE["Mul"]
        return " | ".join(self.parenthesize(arg, PREC) for arg in expr.args)

    def _print_Xor(self, expr):
        """打印异或运算表达式
        
        与其他打印器的差异：
        - polars/pandas: 使用 ^
        - rust: 使用链式 .xor()
        - sql: 使用 XOR
        """
        PREC = PRECEDENCE["Mul"]
        return " ^ ".join(self.parenthesize(arg, PREC) for arg in expr.args)

    def _print_And(self, expr):
        """打印与运算表达式
        
        与其他打印器的差异：
        - polars/pandas: 使用 &
        - rust: 使用链式 .and()
        - sql: 使用 AND
        """
        PREC = PRECEDENCE["Mul"]
        return " & ".join(self.parenthesize(arg, PREC) for arg in expr.args)

    def _print_Not(self, expr):
        """打印非运算表达式
        
        与其他打印器的差异：
        - polars/pandas: 使用 ~
        - rust: 使用 .not()
        - sql: 使用 NOT
        """
        PREC = PRECEDENCE["Mul"]
        return "~%s" % self.parenthesize(expr.args[0], PREC)

    def _print_gp_(self, expr):
        """将 gp_ 函数转换为 cs_ 函数
        
        高级用法：
        1. 丢弃第一个参数（分组键）
        2. 保留其余参数
        3. 函数名去掉 gp_ 前缀，添加 cs_ 前缀
        
        转换示例：
        - gp_mean(x, date, asset) -> cs_mean(x)
        
        注意：
        - 在 polars 中，groupby 操作已经处理了分组键
        - 因此生成的代码中不需要显式传递分组键
        """
        new_args = [self._print(arg) for arg in expr.args[1:]]
        func_name = expr.func.__name__[3:]
        return "cs_%s(%s)" % (func_name, ",".join(new_args))
