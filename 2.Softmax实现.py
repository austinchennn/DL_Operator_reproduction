import math


def softmax(x: list[list[float]]) -> list:
    """

    x: 输入数据 list
    """

    y = []
    for row in x:
        # 稳定性找最大的那个数
        row_max = max(row)
        # 计算这行每个数的e^val
        exp_row = [math.exp(val - row_max) for val in row]
        # 计算分母
        sum_exp = sum(exp_row)
        # 归一化：每个元素除以总和
        probs_row = [val / sum_exp for val in exp_row]
        y.append(probs_row)
    return y





def softmax_not_stablized(x: list) -> list:
    """

    x: 输入数据 list
    """

    y = []
    for row in x:
        # 计算这行每个数的e^val
        exp_row = [math.exp(val) for val in row]
        # 计算分母
        sum_exp = sum(exp_row)
        # 归一化：每个元素除以总和
        probs_row = [val / sum_exp for val in exp_row]
        y.append(probs_row)
    return y

data_list = [[1.0, 2.0, 3.0],
             [9.0, 8.0, 7.0]]
result = softmax(data_list)
print("\n纯 Python Softmax 结果:")
for row in result:
    print(row)
print("-----------------------------------")
result2 = softmax_not_stablized(data_list)
print("\n纯 Python softmax_not_stablized 结果:")
for row in result2:
    print(row)


