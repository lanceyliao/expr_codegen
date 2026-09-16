"""
Python AST 基础用法示例

AST（抽象语法树）是 Python 代码的结构化表示，用于代码分析、转换和生成。
"""

import ast


# ==================== 1. 解析代码为 AST ====================
def example_parse():
    """将代码字符串解析为 AST"""
    code = """
x = 1 + 2
y = x * 3
if x > 0:
    print(y)
"""
    
    # 解析为 AST
    tree = ast.parse(code)
    
    # 查看 AST 结构
    print("AST 结构:")
    print(ast.dump(tree, indent=2))
    
    return tree


# ==================== 2. 遍历 AST ====================
def example_traversal():
    """遍历 AST 节点"""
    code = "x = 1 + 2"
    tree = ast.parse(code)
    
    print("\n=== 遍历所有节点 ===")
    for node in ast.walk(tree):
        print(f"节点类型: {node.__class__.__name__}")
    
    print("\n=== 只遍历子节点 ===")
    for node in ast.iter_child_nodes(tree):
        print(f"子节点类型: {node.__class__.__name__}")


# ==================== 3. NodeVisitor 模式 ====================
class VariableVisitor(ast.NodeVisitor):
    """访问所有变量名"""
    
    def visit_Name(self, node):
        print(f"变量名: {node.id}")
        self.generic_visit(node)


def example_visitor():
    """使用 NodeVisitor 遍历 AST"""
    code = "x = 1 + 2; y = x * 3"
    tree = ast.parse(code)
    
    visitor = VariableVisitor()
    visitor.visit(tree)


# ==================== 4. NodeTransformer 模式 ====================
class ConstantFolder(ast.NodeTransformer):
    """常量折叠：在编译时计算常量表达式"""
    
    def visit_BinOp(self, node):
        # 先递归处理子节点
        self.generic_visit(node)
        
        # 如果左右都是常量，直接计算
        if isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant):
            left_val = node.left.value
            right_val = node.right.value
            
            if isinstance(node.op, ast.Add):
                return ast.Constant(value=left_val + right_val)
            elif isinstance(node.op, ast.Mult):
                return ast.Constant(value=left_val * right_val)
        
        return node


def example_transformer():
    """使用 NodeTransformer 转换 AST"""
    code = "x = 1 + 2 + 3"
    tree = ast.parse(code)
    
    print("\n=== 转换前 ===")
    print(ast.unparse(tree))
    
    transformer = ConstantFolder()
    new_tree = transformer.visit(tree)
    
    print("=== 转换后 ===")
    print(ast.unparse(new_tree))


# ==================== 5. 修改 AST 节点 ====================
def example_modify():
    """修改 AST 节点"""
    code = "x = 1"
    tree = ast.parse(code)
    
    # 找到赋值语句
    assign = tree.body[0]
    
    # 修改赋值右侧
    assign.value = ast.BinOp(
        left=ast.Constant(value=1),
        op=ast.Add(),
        right=ast.Constant(value=2)
    )
    
    print("\n=== 修改后 ===")
    print(ast.unparse(tree))


# ==================== 6. 添加父节点引用 ====================
class ParentSetter(ast.NodeVisitor):
    """为 AST 节点添加父节点引用"""
    
    def visit(self, node):
        for child in ast.iter_child_nodes(node):
            child._parent = node
        self.generic_visit(node)


def example_parent_reference():
    """使用父节点引用"""
    code = "x = 1 + 2"
    tree = ast.parse(code)
    
    # 查找所有 Name 节点
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            parent = getattr(node, '_parent', None)
            print(f"变量 {node.id} 的父节点: {parent.__class__.__name__}")
    
    # 添加父节点引用
    setter = ParentSetter()
    setter.visit(tree)
    print("visit 完成")
    
    # 查找所有 Name 节点
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            parent = getattr(node, '_parent', None)
            print(f"变量 {node.id} 的父节点: {parent.__class__.__name__}")


# ==================== 7. 检查节点类型 ====================
def example_check_types():
    """检查节点类型和属性"""
    code = "x = 1 + 2"
    tree = ast.parse(code)
    
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp):
            print(f"二元运算: {node.op.__class__.__name__}")
            print(f"左操作数: {ast.unparse(node.left)}")
            print(f"右操作数: {ast.unparse(node.right)}")


# ==================== 8. 生成代码 ====================
def example_codegen():
    """从 AST 生成代码"""
    code = "x = 1 + 2"
    tree = ast.parse(code)
    
    # 使用 ast.unparse 生成代码（Python 3.9+）
    generated = ast.unparse(tree)
    print(f"\n生成的代码: {generated}")


# ==================== 9. 处理函数定义 ====================
class FunctionAnalyzer(ast.NodeVisitor):
    """分析函数定义"""
    
    def visit_FunctionDef(self, node):
        print(f"\n函数名: {node.name}")
        print(f"参数: {[arg.arg for arg in node.args.args]}")
        print(f"返回值: {ast.unparse(node.returns) if node.returns else 'None'}")
        self.generic_visit(node)


def example_function_analysis():
    """分析函数定义"""
    code = """
def add(a, b):
    return a + b
"""
    tree = ast.parse(code)
    analyzer = FunctionAnalyzer()
    analyzer.visit(tree)


# ==================== 10. 处理导入语句 ====================
class ImportCollector(ast.NodeVisitor):
    """收集所有导入"""
    
    def __init__(self):
        self.imports = []
    
    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        module = node.module
        for alias in node.names:
            self.imports.append(f"{module}.{alias.name}")
        self.generic_visit(node)


def example_import_analysis():
    """分析导入语句"""
    code = """
import os
import sys
from math import sin, cos
"""
    tree = ast.parse(code)
    collector = ImportCollector()
    collector.visit(tree)
    
    print(f"\n导入列表: {collector.imports}")


# ==================== 运行所有示例 ====================
if __name__ == "__main__":
    print("=" * 50)
    print("Python AST 基础用法示例")
    print("=" * 50)
    
    example_parse()
    example_traversal()
    example_visitor()
    example_transformer()
    example_modify()
    example_parent_reference()
    example_check_types()
    example_codegen()
    example_function_analysis()
    example_import_analysis()
