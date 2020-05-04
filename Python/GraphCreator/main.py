from graphviz import Digraph
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

with open("input.txt", "r") as f:
    #data = [elem.strip().split('->') for elem in f]
    data = [list(map(lambda x: x.strip(), elem.split('->'))) for elem in f]
    x = np.transpose(data)
x.astype(int)

d = {
    1 :'YAHOO',
    2 : 'BING',
    3 : 'EBAY',
    4 : 'IEEE_XPLORE',
    5 : 'EXPEDIA',
    6 : 'ALLEGRO',
    7 : 'WIKIPEDIA',
    8 : 'AMAZON',
    9 : 'YOUTUBE',
    10 : 'GOOGLE',
    11 : 'FACEBOOK'
}

size = (np.size(x,1)-1)
f = Digraph('Rank', filename='rank.gv')
f.attr(rankdir='LR', size='8,5')
f.attr('node', shape='circle')

for i in range(size):
    a = x.item((0, i))
    b = x.item((1, i))
    f.edge(d[int(a)], d[int(b)])
f.view()


