import json
from pprint import pprint

import matplotlib.pyplot as plt
import networkx as nx


def add_edge_to_graph(graph, object1, object2, relation, multi=False):
    if relation == "a on b":
        graph.add_edge(f"{object1['id']}_{object1['object_tag']}", f"{object2['id']}_{object2['object_tag']}", relation="on")
        if multi:
            graph.add_edge(f"{object2['id']}_{object2['object_tag']}", f"{object1['id']}_{object1['object_tag']}", relation="under")
    elif relation == "b on a":
        graph.add_edge(f"{object1['id']}_{object1['object_tag']}", f"{object2['id']}_{object2['object_tag']}", relation="under")
        if multi:
            graph.add_edge(f"{object2['id']}_{object2['object_tag']}", f"{object1['id']}_{object1['object_tag']}", relation="on")
    return graph


def add_object_to_graph(graph, graph_object):
    if (
        graph_object["object_tag"] not in graph.nodes
        and graph_object["object_tag"] != "invalid"
    ):
        graph.add_node(
            f"{graph_object['id']}_{graph_object['object_tag']}", object=graph_object
        )
    return graph


SG_PATH = "/private/home/priparashar/traj3/sg_cache/cfslam_object_relations.json"

scene_graph = json.load(open(SG_PATH, "r"))

# create a dictionary of objects from scene_graph scene_graph is a list of dicts
# new dict has scene_graph[i]["id"] as key and scene_graph[i] as value
pprint(scene_graph)
# nx_graph = nx.MultiDiGraph()
nx_graph = nx.DiGraph()

for edge in scene_graph:
    object1 = edge["object1"]
    nx_graph = add_object_to_graph(nx_graph, object1)
    object2 = edge["object2"]
    nx_graph = add_object_to_graph(nx_graph, object1)
    relation = edge["object_relation"]
    nx_graph = add_edge_to_graph(nx_graph, object1, object2, relation)
    if relation != "none of these" and relation != "invalid" and relation != "FAIL":
        print(f"a:{object1['object_tag']}, b:{object2['object_tag']}, {relation=}")

print(nx_graph.nodes())
print(nx_graph.edges())
nx.draw_networkx(nx_graph, with_labels=True, node_size=1000, font_size=8, node_color="skyblue")
plt.savefig("scene_graph.png")
nx.draw_networkx_edge_labels(nx_graph, pos=nx.spring_layout(nx_graph))
plt.savefig("scene_graph_with_edges.png")
