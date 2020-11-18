#ifndef GRAPH_STRUCT_H
#define GRAPH_STRUCT_H

#include <vector>
#include <map>
#include <iostream>
#include <cassert>
#include <algorithm>

template<typename T>
class LinkedTable   // 用vector封装的连接表，逻辑上就是一个二维数组
{
public:
	LinkedTable();
    ~LinkedTable();
	void AddEntry(int head_id, T content); // 在head_id的行插入内容
	void Resize(int new_n);
	int n;                                 // 相当与stl中的size
	std::vector< std::vector<T> > head;    // 存放数据的实体
private:
	int ncap;  // 类似于vector的capcity()，实际的容量 只增不减 resize中使用
};

class GraphStruct // 表征 有向图 的类
{
public:
	GraphStruct();
	~GraphStruct();
	void AddEdge(int idx, int x, int y);           // idx 边的编号从0开始，x头节点，y尾巴节点
	void AddNode(int subg_id, int n_idx);          // 子图的id（用于mini-batch） n_idx 节点的id
	void Resize(unsigned _num_subgraph, unsigned _num_nodes = 0);
	
	LinkedTable< std::pair<int, int> > *out_edges; // 保存节点的出度点的表指针
	LinkedTable< std::pair<int, int> > *in_edges;  // 保存节点的入度点的表指针
	LinkedTable< int >* subgraph;                  // 指向保存每个子图对应的节点信息
	std::vector< std::pair<int, int> > edge_list;  // 保存边 注意这边不是指针  

	unsigned num_nodes;
	unsigned num_edges;
	unsigned num_subgraph;	
};

extern GraphStruct batch_graph; // global variable // 貌似没有用

#endif