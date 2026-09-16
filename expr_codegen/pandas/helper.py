"""
A、B、C=MACD()无法生成DAG，所以变通的改成

A=unpack(MACD(),0)
B=unpack(MACD(),1)
C=unpack(MACD(),2)

cse能自动提取成

_x_0 = MACD()

但 df['_x_0'] 是无法放入tuple的，所以决定用另一个类来实现兼容

"""
import pandas as pd


class GlobalVariable(object):
    """全局变量容器，兼容 DataFrame 和 tuple 存储
    
    高级用法：
    1. 使用 dict 存储 tuple 值（如 MACD 返回的多个值）
    2. 使用 DataFrame 存储普通 Series
    3. __getitem__ 优先从 dict 读取，再从 DataFrame 读取
    4. __setitem__ 根据 value 类型选择存储位置
    
    应用场景：
    - 解决 A,B,C=MACD() 无法生成 DAG 的问题
    - 将 tuple 存储在 dict 中，Series 存储在 DataFrame 中
    - 避免直接将 tuple 放入 DataFrame 列
    
    注意：
    - tuple 存入 dict 时，在 DataFrame 中占位（False）避免 drop 报错
    """
    def __init__(self):
        self.dict = {}
        self.df = pd.DataFrame()

    def __getitem__(self, item):
        if item in self.dict:
            return self.dict[item]
        return self.df[item]

    def __setitem__(self, key, value):
        if isinstance(value, tuple):
            # tuple存字典中
            self.dict[key] = value
            # 占位，避免drop时报错
            self.df[key] = False
        else:
            self.df[key] = value
