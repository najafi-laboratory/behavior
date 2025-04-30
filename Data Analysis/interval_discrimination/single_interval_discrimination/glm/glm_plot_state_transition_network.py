# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 22:49:14 2025

@author: timst
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def plot_state_transition_network(model, model_summary, mouse_id=None, session_date=None, threshold=0.05, figsize=(7, 6)):
    """
    Finalized network plot of GLM-HMM state transitions with fully offset labels.
    """
    print("🧠 Plotting state transition diagram (with label offset)...")
    glm_hmm = model['glm_hmm']
    num_states = glm_hmm.K
    trans_matrix = glm_hmm.transitions.transition_matrix
    state_labels = [f"State {s['state_id']}: {s['label']}" for s in model_summary["states"]]

    G = nx.DiGraph()
    for i in range(num_states):
        G.add_node(i)

    for i in range(num_states):
        for j in range(num_states):
            prob = trans_matrix[i, j]
            if prob >= threshold:
                G.add_edge(i, j, weight=prob)

    pos = nx.circular_layout(G)

    plt.figure(figsize=figsize)

    # Draw nodes
    node_colors = ['#1f77b4', '#2ca02c', '#d62728'][:num_states]
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color=node_colors, alpha=0.8)

    # Draw arrows
    nx.draw_networkx_edges(
        G, pos,
        arrowstyle='-|>',
        arrowsize=18,
        edge_color='gray',
        width=2,
        connectionstyle='arc3,rad=0.2'
    )

    # Draw edge labels
    edge_weights = nx.get_edge_attributes(G, 'weight')
    edge_labels = {(u, v): f"{w:.2f}" for (u, v), w in edge_weights.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

    # Adjust label positions — push them outward
    # label_radius = 1.45
    label_radius = 1.25
    for i, (x, y) in pos.items():
        angle = np.arctan2(y, x)
        label_x = label_radius * np.cos(angle)
        label_y = label_radius * np.sin(angle)
        ha = 'left' if label_x >= 0 else 'right'
        va = 'bottom' if label_y >= 0 else 'top'
        plt.text(label_x, label_y, state_labels[i], fontsize=10, ha=ha, va=va, wrap=True)

    # Title
    title = f"GLM-HMM State Transition Diagram\n{mouse_id or ''} {session_date or ''}"
    plt.title(title.strip(), fontsize=14, pad=25)

    plt.axis('off')
    plt.tight_layout()
    plt.show()   
    
   
    
   
    
   
    
   
    
   
    # """
    # Improved network plot of GLM-HMM state transitions with clear node/label spacing.
    # """
    # print("🎯 Plotting refined state transition network...")
    # glm_hmm = model['glm_hmm']
    # num_states = glm_hmm.K
    # trans_matrix = glm_hmm.transitions.transition_matrix
    # state_labels = [f"State {s['state_id']}\n{s['label']}" for s in model_summary["states"]]

    # G = nx.DiGraph()
    # for i in range(num_states):
    #     G.add_node(i)

    # for i in range(num_states):
    #     for j in range(num_states):
    #         prob = trans_matrix[i, j]
    #         if prob >= threshold:
    #             G.add_edge(i, j, weight=prob)

    # pos = nx.circular_layout(G)

    # # -- DRAW
    # plt.figure(figsize=figsize)

    # # Draw nodes (smaller)
    # node_colors = ['#1f77b4', '#2ca02c', '#d62728'][:num_states]
    # nx.draw_networkx_nodes(G, pos, node_size=1000, node_color=node_colors, alpha=0.8)

    # # Draw edges
    # nx.draw_networkx_edges(
    #     G, pos,
    #     arrowstyle='-|>',
    #     arrowsize=20,
    #     edge_color='gray',
    #     width=2,
    #     connectionstyle='arc3,rad=0.2'
    # )

    # # Edge labels
    # edge_weights = nx.get_edge_attributes(G, 'weight')
    # edge_labels = {(u, v): f"{w:.2f}" for (u, v), w in edge_weights.items()}
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

    # # External label positions (slightly offset from circle)
    # label_pos = {}
    # for node, (x, y) in pos.items():
    #     angle = np.arctan2(y, x)
    #     label_offset = 1.2  # radial multiplier
    #     label_pos[node] = (x * label_offset, y * label_offset)

    # # Plot labels outside of nodes
    # for i, (x, y) in label_pos.items():
    #     plt.text(x, y, state_labels[i], ha='center', va='center', fontsize=10, wrap=True)

    # # Title
    # title = f"State Transition Diagram\n{mouse_id or ''} {session_date or ''}"
    # plt.title(title.strip(), fontsize=14, pad=30)

    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()    
    
    
    
    
    
    # """
    # Plots a network diagram of state transitions from a GLM-HMM model.

    # Args:
    #     glm_hmm: Trained GLM-HMM model.
    #     model_summary (dict): Output from summarize_glm_hmm_model().
    #     mouse_id (str): Optional mouse identifier for title.
    #     session_date (str): Optional session date for title.
    #     threshold (float): Minimum transition probability to draw edge.
    #     figsize (tuple): Figure size.
    # """
    # print("📈 Building state transition network...")

    # glm_hmm = model['glm_hmm']
    # num_states = glm_hmm.K
    # trans_matrix = glm_hmm.transitions.transition_matrix

    # G = nx.DiGraph()

    # # Use summary labels for nodes
    # state_labels = [f"{s['state_id']} – {s['label']}" for s in model_summary["states"]]

    # for i in range(num_states):
    #     G.add_node(i, label=state_labels[i])

    # # Add edges with weights
    # for i in range(num_states):
    #     for j in range(num_states):
    #         prob = trans_matrix[i, j]
    #         if prob >= threshold:
    #             G.add_edge(i, j, weight=prob)

    # pos = nx.circular_layout(G)

    # edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    # edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}

    # node_colors = ["#1f77b4", "#2ca02c", "#d62728"][:num_states]

    # plt.figure(figsize=figsize)
    # nx.draw_networkx_nodes(G, pos, node_size=1600, node_color=node_colors, alpha=0.9)
    # nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]['label'] for n in G.nodes()}, font_size=9, font_color='white')
    # nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=20, edge_color='gray', width=2, connectionstyle='arc3,rad=0.2')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # title = f"State Transition Network"
    # subtitle = " ".join(filter(None, [mouse_id, session_date]))
    # plt.title(f"{title}\n{subtitle}", fontsize=13, pad=20)

    # plt.axis("off")
    # plt.tight_layout()
    # plt.show()

