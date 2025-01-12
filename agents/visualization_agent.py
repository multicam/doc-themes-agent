# agents/visualization_agent.py
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import List, Dict
from models.theme import Theme
import numpy as np
from urllib.parse import quote

class VisualizationAgent:
    def __init__(self):
        sns.set_style('whitegrid')

    def plot_theme_distribution(self, themes: List[Theme]) -> None:
        """Plot distribution of articles across themes"""
        theme_sizes = [len(theme.articles) for theme in themes]
        theme_labels = [f"Theme {theme.id + 1}\n({', '.join(theme.top_terms[:3])})"
                        for theme in themes]

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(themes)), theme_sizes)
        plt.xticks(range(len(themes)), theme_labels, rotation=45, ha='right')
        plt.title('Distribution of Articles Across Themes')
        plt.ylabel('Number of Articles')
        plt.tight_layout()
        plt.show()

    def plot_theme_network(self, themes: List[Theme],
                           min_edge_weight: float = 0.3) -> None:
        """Plot network of theme relationships"""
        G = nx.Graph()

        # Add nodes
        for theme in themes:
            G.add_node(theme.id,
                       size=len(theme.articles),
                       label=f"Theme {theme.id + 1}")

        # Add edges for similar themes
        for i, theme1 in enumerate(themes):
            for j, theme2 in enumerate(themes[i + 1:], i + 1):
                similarity = 1 - np.linalg.norm(theme1.vector - theme2.vector)
                if similarity > min_edge_weight:
                    G.add_edge(theme1.id, theme2.id, weight=similarity)

        # Plot
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos,
                               node_size=[G.nodes[n]['size'] * 100 for n in G.nodes],
                               node_color='lightblue')

        # Draw edges
        nx.draw_networkx_edges(G, pos,
                               width=[G[u][v]['weight'] * 2 for u, v in G.edges()])

        # Draw labels
        nx.draw_networkx_labels(G, pos,
                                labels=nx.get_node_attributes(G, 'label'))

        plt.title('Theme Relationship Network')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def generate_html_report(self, themes: List[Theme]) -> str:
        """Generate HTML report of themes"""
        html = """
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .theme { margin-bottom: 20px; padding: 10px; border: 1px solid #ddd; }
                .terms { color: #666; }
                .articles { margin-left: 20px; }
            </style>
        </head>
        <body>
            <h1>Theme Analysis Report</h1>
        """

        for theme in themes:
            html += f"""
            <div class="theme">
                <h2>Theme {theme.id + 1}</h2>
                <div class="terms">
                    <b>Key terms:</b> {', '.join(f'{term} ({weight:.2f})'
                                                 for term, weight in theme.terms[:5])}
                </div>
                <p><b>Coherence:</b> {theme.coherence:.3f}</p>
                <p><b>Number of articles:</b> {len(theme.articles)}</p>
                <details class="articles">
                    <summary><b>The articles:</b></summary>
                    <ul>
            """

            for article in sorted(theme.articles,
                                  key=lambda x: x['score'],
                                  reverse=True):
                if article['score'] < 0.01:
                    break

                html += f"""
                    <li style="padding: .5em">
                        <a href="obsidian://open?file={quote(article['path'])}">{article['path']} (score: {article['score']:.2f})</a>
                    </li>
                """

            html += """
                    </ul>
                </details>
            </div>
            """

        html += """
        </body>
        </html>
        """

        return html