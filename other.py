import numpy as np


def recomender(paperlist, colum, history, num):
    if colum == 'mix':
        score = np.dot(paperlist.iloc[:, 1:4], [1, 1, 1])
    else:
        score = paperlist['{}'.format(colum)]

    paperlist['score'] = score
    docsn = paperlist.sort_values(by='score')
    docs_r = docsn["id"]

    for x in history:
        docs_r = docs_r.drop(x - 1)
    docs_r = [x + 1 for x in docs_r[0:num]]
    return docs_r


def select_num(docs_r, docs_u):
    count = 0
    docs_u = list(docs_u)
    for id in docs_r:
        if id in docs_u:
            count = count + 1
    return count


def select_paper(docs_r, docs_u):
    paper = []
    docs_u = list(docs_u)
    for id in docs_r:
        if id in docs_u:
            paper.append(id)
    return paper


def re_se(paperlist, colum, history, num, docs_u):
    docs_r = recomender(paperlist, colum, history, num)
    count = select_num(docs_r, docs_u)
    return count
