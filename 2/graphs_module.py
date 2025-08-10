import matplotlib.pyplot as plt
import numpy as np

class Graph:
    def __init__(self, bidirected=False):
        self.nodes = []                        # nodes vector with some added information 
        self.edges = np.array([], dtype=int)   # adjacency matrix
        self.bidirected = bidirected           # flag variable for bidirected graphs 

        return None

    def __len__(self):
        return len(self.nodes)

    def add_node(self, some_info):
        n0 = self.__len__()                    
        self.nodes.append(some_info)           # actual addition 
        
        n_new = self.__len__()                 # reshape of previous edges variable 
        if self.bidirected:   
            new_edges = np.eye(n_new, n_new, dtype=int)
        else:
            new_edges = np.zeros((n_new, n_new), dtype=int)
        new_edges[:n0, :n0] = self.edges      # coping of old edges variable to the new one  
        self.edges = new_edges
        
        return None
    
    def add_edge(self, i, j):                 # addition of i -> j link 
        self.edges[i, j] = 1                  # addition of j -> i in case of bidirected graphs 
        if self.bidirected:
            self.edges[j, i] = 1
    
        return None
    
    def randomize(self, size=10, bidirected=False):
        if size == -1:                       # -1 corresponds to random graph size  
            size = np.random.randint(1, 20)  

        self.nodes = [i for i in range(size)]
        self.edges = np.random.rand(size, size).round()
        if not bidirected:
            np.fill_diagonal(self.edges, 0)

        return None
    
    def plot(self, fig=None, ax=None, figsize=(8, 8), span=0.05):
        if fig == ax == None:
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_aspect('equal')
            ax.axis('off')

        n_nodes = self.__len__()
        angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
        positions = np.array([(np.cos(a), np.sin(a)) for a in angles])

        for i, (x, y) in enumerate(positions):                               # plot of nodes   
            ax.plot(x, y, 'o', markersize=15, color='skyblue') 
            ax.text(x, y, str(i), fontsize=12, ha='center', va='center')

        for i in range(n_nodes):
            for j in range(n_nodes):
                if self.edges[i, j] == 1:
                    if i == j:                                               # plot of self-tangles  
                        x, y = positions[i]
                        circle = plt.Circle((x, y + 0.1),
                                            0.1,
                                            fill=False,
                                            edgecolor='gray',
                                            linestyle='dashed'
                        )
                        ax.add_patch(circle)

                    else:
                        i_pos = positions[i]                                  # plot of connections with a small span from arrows to nodes
                        j_pos = positions[j]                                  
                        dir = (j_pos - i_pos) / np.linalg.norm(j_pos - i_pos)

                        ax.annotate("",
                            xy=j_pos - span * dir,
                            xytext=i_pos + span * dir,
                            arrowprops=dict(arrowstyle="->", color='gray', lw=1.5),
                        )

        return fig, ax