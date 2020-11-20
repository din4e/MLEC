import tensorflow as tf
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from FINDER import FINDER

# x=set([1,2,3])
# y=set([2,4,5])
# print(x^y)
# print(set([1,2,3])^set([2,4,5]))
# tf.print((tf.ones((3,3))))

# networkx
g = nx.erdos_renyi_graph(n=10, p=0.3)
nx.draw(g)
plt.show()

# ones=tf.Variable(tf.ones([3,3]))
# inter_sess=tf.InteractiveSession()
# ones.initializer.run()
# print(inter_sess.run(ones))