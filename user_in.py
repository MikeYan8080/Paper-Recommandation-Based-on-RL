import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from intro_paper import  sort_mat


# 运用矩阵分解的方法得到用户相似矩阵和论文相似矩阵
def dis_cf_matrix(data):
    row = np.array(data['user.id'])
    col = np.array(data['doc.id'])
    mat = np.zeros((np.max(row), np.max(col)))
    for i in range(len(row)):
        mat[row[i]-1, col[i]-1] = 1

    r = 4
    u, s, v = np.linalg.svd(mat)
    u, s, v = u[:, : r], s[:r], v[:r]
    sk = np.diag(np.sqrt(s))  # r*r
    uk = u @ sk  # m*r
    vk = sk @ v  # r*n

    dis_uk = pdist(uk, metric='euclidean')
    dis_vk = pdist(vk.T, metric='euclidean')
    dis_uk = squareform(dis_uk)
    dis_vk = squareform(dis_vk)
    return dis_uk, dis_vk


# 根据给出论文（读者）矩阵算出所有论文（读者）距离
def con_dis(dis_mat, doc_id):
    li = []
    for x in doc_id:
        li.append(dis_mat[x-1])
    li2 = [np.mean(x) for x in np.transpose(li)]  # 计算平均的距离
    # 将自身距离设置为1，避免推荐自己
    for x in doc_id:
        li2[x-1] = 1
        return li2


# 通过用户名返回论文记录
def user_paper(user_id, data):
    user_data = data.loc[data['user.id'] == user_id]
    return user_data['doc.id']
def find_suit_user(num, data):
    user = []
    user_num = np.max(data['user.id'])
    for x in range(user_num):
        length = len(user_paper(x+1, data))
        if length > num:
            user.append(x+1)
return user
