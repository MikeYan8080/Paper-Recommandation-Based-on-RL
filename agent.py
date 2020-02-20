import numpy as np
# 用户从推荐的论文中挑选喜欢的
def user(docs_r, docs_u):
    docs_s = []
    docs_u = list(docs_u)
    for id in docs_r:
        if id in docs_u:
            docs_s.append(id)
    # print(len(docs_s))
    return docs_s
# 通过选择的论文确定状态
def state(docs_s, paper_list):
    if docs_s:
        id = [x-1 for x in docs_s]
        score = [np.array(paper_list[x:x+1])[0][1:4] for x in id]
        ss = np.sum(score, axis=0)
        if ss[2] <= ss[1] & ss[1] <= ss[1]:
            s = 1
        elif ss[0] <= ss[2] & ss[2] <= ss[1]:
            s = 2
        elif ss[1] <= ss[0] & ss[0] <= ss[2]:
            s = 3
        elif ss[1] <= ss[2] & ss[2] <= ss[0]:
            s = 4
        elif ss[0] <= ss[1] & ss[1] <= ss[1]:
            s = 5
        else:
            s = 6
    else:
        s = 0
    return s
# 更新Q表
def renewQ(s, a, r, Q):
    Q[s, a] += r - 0.5
    return Q
#根据状态和Q表给出action
def agent(s, Q, e=0.2):
    x = np.random.random()
    if x > e:
        a = np.argmax(Q[s])
    else:
        a = np.random.randint(0, 7)
    return a
# 根据action推荐
def recomender(a, action, docs, history, m=10):
    score = np.dot(docs.iloc[:, 1:], action[a])
    docs['score'] = score
    docsn = docs.sort_values(by='score')
    docs_r = docsn["id"]
    for x in history:
        docs_r = docs_r.drop(x-1)
    docs_r = [x+1 for x in docs_r[0:m]]
    return docs_r
