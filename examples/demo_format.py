"""
代码块的可变参数
"""

from expr_codegen import code_format, codegen_exec


def _code_block_1(length):
    # 符合语法，但无法实现动态变量名
    SMA = ts_mean(CLOSE, length)
    STD = ts_std_dev(CLOSE, length)
    TEST = length(CLOSE, length, length * 2)


def _code_block_2(length):
    # 动态变量名
    return f"""
SMA_{length}  = ts_mean(CLOSE, {length})
STD_{length}  = ts_std_dev(CLOSE, {length})
TEST = length(CLOSE, {length}, {length} * 2)
""".format(length=length)


print(code_format(_code_block_1, length=20))
print(_code_block_2(length=20))

df = codegen_exec(None, code_format(_code_block_1, length=20), _code_block_2(length=20), over_null='partition_by')
print(df)
