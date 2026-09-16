import inspect

from sympy import Basic, Function, StrPrinter


# TODO: 如有新添加函数，但表达式有变更才需要在此补充对应的打印代码，否则可以省略

class RustStrPrinter(StrPrinter):
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

                # 所有以gp_开头的函数都转换成cs_开头
                if printmethodname.startswith('_print_gp_'):
                    printmethodname = "_print_gp_"

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
        - 使用 col("expr.name") 访问列
        - 保留字（_NONE_, _TRUE_, _FALSE_）直接返回
        
        注意：
        - 这种语法适合 Rust polars 的列访问
        """
        if expr.name in ('_NONE_', '_TRUE_', '_FALSE_'):
            return expr.name
        return f'col("{expr.name}")'

    def _print_Or(self, expr):
        """打印或运算表达式
        
        与 polars 的差异：
        - 使用链式 .or() 调用
        - 使用 .join() 连接多个参数
        - [2:] 去掉开头的 "."
        
        转换示例：
        - a | b | c -> a.or(b).or(c)
        """
        new_args = [f"or({self._print(arg)})" for arg in expr.args]
        return ".".join(new_args)[2:]

    def _print_Xor(self, expr):
        """打印异或运算表达式
        
        与 polars 的差异：
        - 使用链式 .xor() 调用
        - [3:] 去掉开头的 ".x"
        """
        new_args = [f"xor({self._print(arg)})" for arg in expr.args]
        return ".".join(new_args)[3:]

    def _print_And(self, expr):
        """打印与运算表达式
        
        与 polars 的差异：
        - 使用链式 .and() 调用
        """
        new_args = [f"and({self._print(arg)})" for arg in expr.args]
        return ".".join(new_args)[3:]

    def _print_Not(self, expr):
        """打印非运算表达式
        
        与 polars 的差异：
        - 使用 .not() 方法
        """
        return "(%s).not()" % self._print(expr.args[0])

    def _print_gp_(self, expr):
        """gp_函数都转换成cs_函数，但要丢弃第一个参数"""
        new_args = [self._print(arg) for arg in expr.args[1:]]
        func_name = expr.func.__name__[3:]
        return "cs_%s(%s)" % (func_name, ",".join(new_args))

    def _print_Integer(self, expr):
        """打印整数
        
        高级用法：
        1. inspect.stack() 获取调用栈
        2. 检查调用者是否为特定函数
        3. 在特定上下文中使用 lit() 包装
        
        注意：
        - 在 Pow、Add、Mul、Relational 中使用 lit() 包装
        - 其他情况下直接返回数字
        - 这是 Rust polars 的类型要求
        """
        caller_frame = inspect.stack()[2]
        caller_name = caller_frame.function
        if caller_name in ("_print_Pow", "_print_Add", "_print_Mul", "_print_Relational"):
            return "lit(%s)" % super()._print_Integer(expr)
        else:
            return super()._print_Integer(expr)

    def _print_Float(self, expr):
        """打印浮点数
        
        高级用法：
        1. 与 _print_Integer 类似，使用 inspect.stack 检查调用者
        2. 在特定上下文中使用 lit() 包装
        
        注意：
        - 在 Pow、Add、Mul、Relational 中使用 lit() 包装
        - 这是 Rust polars 的类型要求
        """
        caller_frame = inspect.stack()[2]
        caller_name = caller_frame.function
        if caller_name in ("_print_Pow", "_print_Add", "_print_Mul", "_print_Relational"):
            return "lit(%s)" % super()._print_Float(expr)
        else:
            return super()._print_Float(expr)

    def _print_Relational(self, expr):
        """打印关系表达式（比较运算）
        
        高级用法：
        1. 使用 charmap 映射 Python 比较符到 Rust 方法
        2. 使用链式调用格式
        
        映射关系：
        - == -> eq
        - != -> neq
        - < -> lt
        - > -> gt
        - >= -> gt_eq
        - <= -> lt_eq
        
        转换示例：
        - a == b -> a.eq(b)
        - a < b -> a.lt(b)
        """

        charmap = {
            "==": "eq",
            "!=": "neq",
            "<": "lt",
            ">": "gt",
            ">=": "gt_eq",
            "<=": "lt_eq",
        }

        if expr.rel_op in charmap:
            return '(%s).%s(%s)' % (self._print(expr.lhs), charmap[expr.rel_op], self._print(expr.rhs))

        return super()._print_Relational(expr)
