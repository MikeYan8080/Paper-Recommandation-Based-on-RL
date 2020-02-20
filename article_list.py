import pandas as pd
import numpy as np
from intro_paper import con_sim_matrix, con_sim, sort_mat
from user_in import dis_cf_matrix, con_dis
# 输入各种相关性矩阵，和要比对的文章，输出每篇文章的得分
def art_array(mat1, mat2, mat3, docs):
    x1 = con_sim(mat1, docs)
    x1 = sort_mat(x1)
    x2 = con_sim(mat2, docs)
    x2 = sort_mat(x2)
    x3 = con_dis(mat3, docs)
    x3 = sort_mat(x3, ascend=True)
    paper_list = pd.DataFrame({'id': range(len(x1)), 'x1': x1, 'x2': x2, 'x3': x3})
return paper_list
