import numpy as np
import random
import igraph as ig
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

graph = ig.Graph.Read_GML('karate.gml')  # Karate Club
G = nx.read_gml('karate.gml', label='id')
M = graph.vcount()  # m=34
N = graph.ecount()  # n=78
edgelist = graph.get_edgelist()  # 边的list
neighbors = graph.neighborhood()  # 与各个点相连的其他点的集合
k = graph.degree()  # 各个点的度
Q_list = []


# 邻接矩阵
def compute_A(m):
    A = np.zeros((m, m), dtype=np.int)
    for i in range(m):
        for j in range(m):
            if i is j:
                break
            # 由邻节点构建邻接矩阵
            if j in neighbors[i]:
                A[i][j] = 1
                A[j][i] = 1
    return A


# 相似度矩阵
def similarity(m):
    similar = np.zeros((34, 34))
    for i in range(34):
        for j in range(34):
            p = []
            q = []
            for k in range(34):
                if (i in neighbors[k] and j in neighbors[k]):
                    p.append(k)
                if (i in neighbors[k] or j in neighbors[k]):
                    q.append(k)
            l1 = len(p)
            l2 = len(q)
            similar[i][j] = l1 / l2
            similar[j][i] = l1 / l2
    for k in range(m):
        similar[k][k] = 0
    return similar


def init_compute_clusters(clusters, m):
    for i in range(m):
        clusters.append(-1)
    return clusters


# 模块度Q计量
def compute_Q(f, adjacency):
    Q = 0
    for i in range(M):
        for j in range(M):
            if f[i] == f[j]:
                Q += (adjacency[i][j] - (k[i] * k[j] / (2 * N))) / (2 * N)
            else:
                Q += 0
    Q_list.append(Q)
    return Q


def cluster(sim):
    m = len(sim)
    max_s = 0
    x = 0
    y = 0
    for i in range(m):
        for j in range(m):
            if (sim[i][j] > max_s):
                x = i
                y = j
                max_s = sim[i][j]
    for i in range(m):
        sim[x][i] = min(sim[x][i], sim[y][i])
        # sim[x][i] = max(sim[x][i], sim[y][i])
        sim[y][i] = sim[x][i]
    return x, y


def outputData(clusters, edges):  # 输出无向图的连接关系，输出分类结果
    edge_out = []
    graph_file = 'grafh.csv'
    name1 = ['nodea', 'edgetype', 'nodeb']
    for edge in edges:
        edge_out.append((edge[0] + 1, 1, edge[1] + 1))
    out1 = pd.DataFrame(columns=name1, data=edge_out)
    out1.to_csv(graph_file)
    attribute_file = 'class.csv'
    name2 = ['node', 'class']
    attribute_out = []
    for i in range(1, M + 1):
        attribute_out.append((i, clusters[i - 1]))
    out2 = pd.DataFrame(columns=name2, data=attribute_out)
    out2.to_csv(attribute_file)


def draw():
    plt.figure()
    plt.plot(range(len(Q_list)), Q_list, color="b", linewidth=1)
    plt.xlabel('merge times')
    plt.ylabel("Q")
    plt.scatter(range(len(Q_list)), Q_list, linewidths=3, s=2, c='b')
    plt.savefig(fname="3.png")
    plt.show()


def getrandomcolor():
    colorarr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorarr[random.randint(0, 14)]
    return "#" + color


if __name__ == '__main__':
    A = compute_A(M)
    similar = similarity(M)
    clusters = []
    cluster_complete = []
    
    clusters = init_compute_clusters(clusters, M)
    i = 0
    flag = 0
    Q = 0
    q_max = -1
    while (i < 100):
        x, y = cluster(similar)
        if clusters[x] != -1 and clusters[y] != -1:
            temp = clusters[y]
            for j in range(M):
                if clusters[j] == temp:
                    clusters[j] = clusters[x]
        if clusters[x] == -1 or clusters[y] == -1:  # 两个簇中至少有一个未分类
            if clusters[x] == -1 and clusters[y] == -1:  # 两个簇都未分类
                flag += 1  # 新建一个分类保存此簇
                clusters[x] = clusters[y] = flag
            elif clusters[x] == -1:  # x未分类y已分类，将x并到y上去
                clusters[x] = clusters[y]
            elif clusters[y] == -1:  # x已分类y未分类，将y并到x上去
                clusters[y] = clusters[x]
        Q = compute_Q(clusters, A)
        print(Q)
        if (Q > q_max):
            outputData(clusters, edgelist)
            q_max = Q
            over_clusters = {}
            for i in range(1, M + 1):
                over_clusters[i] = clusters[i - 1]
        i += 1
    
    pos = nx.spring_layout(G)
    count = 0.
    
    list_nodes = []
    
    for i in range(1, M):
        list_node = []
        for j in range(1, M):
            if over_clusters[j] == i:
                list_node.append(j)
        if (len(list_node) != 0):
            list_nodes.append(list_node)
    
    print(q_max)
    
    for list_node in list_nodes:
        print(list_node)
    for x in list_nodes:
        nx.draw(G, pos=pos, nodelist=x, node_color=getrandomcolor(), with_labels=True)
    plt.savefig(fname="4.png")
    plt.show()
    
    draw()
