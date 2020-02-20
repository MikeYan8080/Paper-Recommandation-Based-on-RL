import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


#计算相似度
def con_sim_matrix(data):
    #tf-idf矩阵
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(data)
    #print(tfidf_matrix.toarray())
    #相似度矩阵
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim


#截取出输入的doc_id中论文的相似度列表
def con_sim(matrix, doc_id):
    li = []
    for x in doc_id:
       li.append(matrix[x-1])
    li2 = [np.mean(x) for x in np.transpose(li)]#计算平均的相似度
    #将自身相似度设置为0，避免推荐自己
    for x in doc_id:
        li2[x-1] = 0
    return li2


#p排序并标注秩
def sort_mat(sim, ascend=False):
    dd = pd.DataFrame({"x": sim, "y": range(len(sim))})
    dd = dd.sort_values(by="x", ascending=ascend)
    dd['z'] = range(len(sim))
    dd = dd.sort_values(by="y")
    return dd["z"]

3、论文-用户特征导入包 user_in.py
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
