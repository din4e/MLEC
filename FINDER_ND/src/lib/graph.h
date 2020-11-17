#ifndef GRAPH_H
#define GRAPH_H

#include <map>
#include <vector>
#include <memory>
#include <algorithm>
#include <set>
class Graph
{
public:
    Graph();
    Graph(const int _num_nodes, const int _num_edges, const int* edges_from, const int* edges_to);
    ~Graph();
    int num_nodes;
    int num_edges;
    std::vector<std::vector<int>>                     adj_list; // 邻接表
    std::vector<std::pair<int, int>>                 edge_list; // 边
    double getTwoRankNeighborsRatio(std::vector<int> &covered); // TODO:不在covered中点的交接数
};

class GSet
{
public:
    GSet();
    ~GSet();
    void InsertGraph(int gid, std::shared_ptr<Graph> graph);  // 插入一个Graph实例指针
    std::shared_ptr<Graph> Sample();                          // 随机获取一个Graph实例指针
    std::shared_ptr<Graph> Get(int gid);                      // 获得gid的Graph实例指针
    void Clear();
    std::map<int, std::shared_ptr<Graph> > graph_pool;        // id->Graph指针的映射表
};

extern GSet GSetTrain;                                        // 训练集
extern GSet GSetTest;                                         // 测试集

#endif