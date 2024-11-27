import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV into a DataFrame
data = pd.read_csv('Interaction.csv')
# 创建一个有权重的图
G = nx.Graph()

# 添加边并设置权重
for _, row in data.iterrows():
    G.add_edge(row['source'], row['target'], weight=row['weight']
               )
# 设置节点位置
pos = nx.spring_layout(G, k=0.5, iterations=90)

# Get node degrees (number of edges connected to each node)
node_degrees = dict(G.degree())
# Normalize degrees for color mapping
max_degree = max(node_degrees.values())  # Max degree for scaling
min_degree = min(node_degrees.values())  # Min degree for scaling
# Define colormap (using viridis here, but you can choose other colormaps like 'plasma', 'inferno', etc.)
cmap = plt.get_cmap('viridis')
max_color_range = 0.85
# Map the node degree to a color
node_colors = [cmap((node_degrees[node]-min_degree)/(max_degree-min_degree)*0.6+0.4) for node in G.nodes()]

# 提取边的权重
edge_weights = nx.get_edge_attributes(G, 'weight')
max_weight = max(edge_weights.values())
min_weight = min(edge_weights.values())
edge_widths = [3 + (edge_weights[edge] - min_weight) * 2 / (max_weight - min_weight) for edge in G.edges()]

# 绘制节点
plt.figure(figsize=(12, 10), dpi=600)
nx.draw(G, pos, with_labels=True, node_color=node_colors, width=edge_widths, node_size=1500, font_size=10, font_weight="bold")
plt.savefig('PPI.png')

# Sort nodes by degree in descending order
sorted_node_degrees = sorted(node_degrees.items(), key=lambda item: item[1], reverse=True)
# Output the sorted list of nodes and their degree
for node, degree in sorted_node_degrees:
    print(f"Node: {node}, Degree: {degree}")

