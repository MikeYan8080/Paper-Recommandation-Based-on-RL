import csv
import time
import agent as ag
import pandas as pd
import numpy as np
import other as ot
from intro_paper import con_sim_matrix
from user_in import dis_cf_matrix, user_paper, find_suit_user
from article_list import art_array

Q = np.random.random((7, 7))  # Q表初始化 initial the Q table
# 给action表赋值 initial the Action table
action = [[1, 1, 1], [1, 2, 3],
          [1, 3, 2], [2, 1, 3],
          [2, 3, 1], [3, 1, 2],
          [3, 2, 1]]

if __name__ == "__main__":
    time_start = time.time()
    filename = r'C:\Users\严书航\Desktop\data.csv'
    data = []
    with open(filename, "r", encoding='utf-8', errors='ignore') as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)
        for row in csv_reader:
            data.append(row)

    data = np.array(data)
    data1 = data[:, 1]  # 标题数据 titles
    data2 = data[:, 4]  # 摘要数据 abstracts
    data3 = pd.read_csv("user-info.csv", delimiter=',')
    title_mat = con_sim_matrix(data1)
    abs_mat = con_sim_matrix(data2)
    user_mat, item_mat = dis_cf_matrix(data3)

    time_end = time.time()
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print('time cost: ', time_end - time_start, 's')

    user = find_suit_user(200, data3)  # 挑选样本用户 select the sample users
    print(len(user))

    number = 20  # 每次推荐文章数 the number of paper recommand everytime
    turn = 25  # 推荐次数 the recommendation times
    recall_list = np.zeros((1, turn))
    precision_list = np.zeros((1, turn))
    out = pd.DataFrame()

    for x in user:
        docs_all = user_paper(x, data3)    # 提取读者所有文章 get the paper that the user has read
        doc_history = list(docs_all[:number])  # 选择作为历史数据的部分 choose part of them as historial data
        len_doc_all = len(docs_all)
        paperlist = art_array(title_mat, abs_mat, item_mat, doc_history)  # 根据用户历史计算得到的各论文得分 get the score of paper by historical data

        doc_r = ag.recomender(a=0, action=action, docs=paperlist, history=doc_history)  # 用平均分方法得到第一次推荐 achieve the first recommandation by mean
        a = 0  # action初始值 the initial number of Action
        count = 0  # 推荐成功篇数计数 counter for the number of successful recomandation
        recall_user = []  # 初始化该用户召回率 initialize the recall rate
        precision_user = []  # 初始化该用户准确率 initialize the accuracy rate
        doc_except = list(doc_history)

        for i in range(turn):
            doc_s = ag.user(doc_r, docs_all)   # 用户选择论文 user choose the paper
            #print(doc_s)
            state = ag.state(doc_s, paperlist)   # 选择的论文状态 the chosen paper's status
            reply = len(doc_s)   # 选择的论文数 the number of chosen paper
            count += reply 
            recall = count/len_doc_all
            precision = count/((i+1)*number)
            recall_user.append(recall)
            precision_user.append(precision)

            Q = ag.renewQ(state, a, reply, Q)   # 更新Q表 renew the Q table

            if len(doc_s) != 0: 
                doc_history = doc_history + doc_s  # 更新用户喜爱的论文 renew the paper that the user likes

            doc_except = doc_except + list(doc_r)  # 不推荐的文章集合 the collection of paper that user dislikes

            paperlist = art_array(title_mat, abs_mat, item_mat, doc_history[-number:])  # 根据用户历史计算得到的各论文得分 calculate the score
            a = ag.agent(state, Q)
            doc_r = ag.recomender(a=a, action=action, docs=paperlist, history=doc_except, m=number)

        recall_list = recall_list + np.array(recall_user)
        precision_list = precision_list + np.array(precision_user)
        # print(recall_list)
        time_end = time.time()
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        print('time cost: ', time_end - time_start, 's')

    recall_list = recall_list[0] / len(user)
    precision_list = precision_list[0] / len(user)
    out['recall'] = recall_list
    out['pre'] = precision_list
    print(Q)

    # 参照实验 control experiment
    colum = ['x1', 'x2', 'x3', 'mix']
    for z in colum:
        recall_list = np.zeros((1, turn))
        precision_list = np.zeros((1, turn))
        for x in user:
            docs_all = user_paper(x, data3)  # 提取读者所有文章 get the paper
            doc_history = list(docs_all[:number])  # 选择作为历史数据的部分 select historical part
            len_doc_all = len(docs_all)
            paperlist = art_array(title_mat, abs_mat, item_mat, doc_history)  # 根据用户历史计算得到的各论文得分 caculate the score
            recall_user = []  # 初始化该用户召回率 initialize the recall rate
            precision_user = []  # 初始化该用户准确率 initialize the accuracy rate

            for j in range(turn):
                num = number * (j + 1)
                doc_rr = ot.recomender(paperlist, z, doc_history, num)
                sn = ot.select_num(doc_rr, docs_all)
                recall = sn / len_doc_all
                precision = sn / num
                recall_user.append(recall)
                precision_user.append(precision)

            recall_list = recall_list + np.array(recall_user)
            precision_list = precision_list + np.array(precision_user)

        recall_list = recall_list[0] / len(user)
        precision_list = precision_list[0] / len(user)
        out['recall_{}'.format(z)] = recall_list
        out['pre_{}'.format(z)] = precision_list

out.to_csv('pre_recall.csv')  # 输出结果 output the results
