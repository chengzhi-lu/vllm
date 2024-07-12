import numpy as np
import timeit

# 测试数据
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

# 生成对象列表
objects_list = [Person('Alice', 25), Person('Bob', 30)] * 10000  # 生成 20000 个对象实例

# 使用 np.unique 测试
def test_np_unique():
    np.unique(objects_list)

# 将列表转换为集合测试
def test_list_to_set():
    set(objects_list)

# 测试函数执行时间
time_np_unique = timeit.timeit(test_np_unique, number=10)
time_list_to_set = timeit.timeit(test_list_to_set, number=10)

# 输出结果
print(f"Time taken by np.unique: {time_np_unique:.6f} seconds")
print(f"Time taken by list to set conversion: {time_list_to_set:.6f} seconds")

