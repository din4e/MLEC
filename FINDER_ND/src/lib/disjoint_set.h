#ifndef DISJOINT_SET_H
#define DISJOINT_SET_H

#include <map>
#include <vector>
#include <memory>
#include <algorithm>
#include <set>

// rank优化的并查集
class Disjoint_Set
{
public:
	std::vector<int> unionSet;  // 保存父节点信息 
	std::vector<int> rankCount; // rank优化的并查集，rank保存的节点数
	int maxRankCount;

    Disjoint_Set();
    Disjoint_Set(int graphSize);
    ~Disjoint_Set();

    int                           findRoot(int node);
    void                             merge(int node1, int node2);
    double getBiggestComponentCurrentRatio() const;                // 最大连通分量节点数/节点数 
    int                            getRank(int rootNode) const;
};

#endif