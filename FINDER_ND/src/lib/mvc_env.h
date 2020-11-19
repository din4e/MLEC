#ifndef MVC_ENV_H
#define MVC_ENV_H

#include <vector>
#include <set>
#include <memory>
#include "graph.h"
#include "disjoint_set.h"

class MvcEnv
{
public:
    MvcEnv(double _norm);
    ~MvcEnv();

    void                s0(std::shared_ptr<Graph> _g);
    double            step(int a);
    void stepWithoutReward(int a);

    std::vector<double> Betweenness(std::vector<std::vector<int>> adj_list);

    int randomAction();
    int betweenAction();
    bool isTerminal();

//    double getReward(double oldCcNum);
    double                   getReward();
    double     getMaxConnectedNodesNum();
    double getNumofConnectedComponents();
    void                    printGraph();
    
    double                                CcNum;
    double                                 norm; 
    std::shared_ptr<Graph>                graph;
    std::vector<std::vector<int>>     state_seq; // 每一step的 action_list
    std::vector<int>                    act_seq; 
    std::vector<int>                action_list;

    std::vector<double>              reward_seq; // 
    std::vector<double>             sum_rewards; //

    int                         numCoveredEdges;
    std::set<int>                   covered_set;
    std::vector<int>                 avail_list; // 可以操作的节点
    std::vector<int >              node_degrees;
    int                           total_degrees;
};

#endif