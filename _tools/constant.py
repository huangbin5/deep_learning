# {数据集名称, (数据集URL, sha-1密钥)}
DATA_HUB = dict()
# 数据集托管地址
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'


def count_pos_color_ham(pos=9, color=6, ham=4):
    """
    默认计算6种图案、汉明距为4的代码字数量
    """
    print('{} positions, {} colors, {} differences'.format(pos, color, ham))
    solution = set()
    cur = '1' * pos
    while cur != str(color) * pos:
        for ele in solution:
            dis = 0
            for i in range(pos):
                dis += (cur[i] != ele[i])
            if dis < ham:
                break
        else:
            solution.add(cur)
            print('solution[{}] = {}'.format(len(solution), cur))
        cur_list = list(map(int, cur))
        for i in range(pos - 1, -1, -1):
            cur_list[i] += 1
            if cur_list[i] <= color:
                break
            cur_list[i] = 1
        cur = ''.join(list(map(str, cur_list)))
    return len(solution)
