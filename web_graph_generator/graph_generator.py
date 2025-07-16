"""
Graph generation utilities for web link analysis.

This module provides functionality to create NetworkX graphs from web link data.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx


class GraphGenerator:
    """Generates NetworkX graph from link data."""
    
    def __init__(self, allow_cycles: bool = True):
        """
        Initialize graph generator.
        
        Args:
            allow_cycles: Whether to allow cycles in the graph
        """
        self.allow_cycles = allow_cycles
        self.graph_stats = {
            'nodes_added': 0,
            'edges_added': 0,
            'edges_skipped': 0,
            'cycles_detected': 0
        }
    
    def create_graph(self, graph_data: Dict[str, List[str]]) -> nx.DiGraph:
        """
        Create NetworkX directed graph from link data.
        
        Args:
            graph_data: Dictionary mapping URLs to lists of linked URLs
            
        Returns:
            NetworkX DiGraph with weighted edges
        """
        G = nx.DiGraph()
        
        # Add all nodes first
        for url in graph_data.keys():
            G.add_node(url)
            self.graph_stats['nodes_added'] += 1
        
        # Add edges with weights
        for source_url, target_urls in graph_data.items():
            # Count occurrences of each target URL (weight = number of links)
            target_counts = {}
            for target_url in target_urls:
                target_counts[target_url] = target_counts.get(target_url, 0) + 1
            
            # Add edges with weights
            for target_url, weight in target_counts.items():
                if target_url in graph_data:  # Only add edges to known nodes
                    if not self.allow_cycles:
                        # Check if adding this edge would create a cycle
                        if nx.has_path(G, target_url, source_url):
                            self.graph_stats['cycles_detected'] += 1
                            self.graph_stats['edges_skipped'] += 1
                            logging.debug(f"Skipping edge {source_url} -> {target_url} (would create cycle)")
                            continue
                    
                    G.add_edge(source_url, target_url, weight=weight)
                    self.graph_stats['edges_added'] += 1
                else:
                    # Target URL not in our scraped data
                    self.graph_stats['edges_skipped'] += 1
        
        self._log_graph_statistics(G)
        return G
    
    def _log_graph_statistics(self, graph: nx.DiGraph) -> None:
        """Log graph creation statistics."""
        logging.info(f"Graph created successfully!")
        logging.info(f"  Nodes: {graph.number_of_nodes()}")
        logging.info(f"  Edges: {graph.number_of_edges()}")
        logging.info(f"  Edges skipped: {self.graph_stats['edges_skipped']}")
        
        if not self.allow_cycles:
            logging.info(f"  Cycles detected and prevented: {self.graph_stats['cycles_detected']}")
    
    def analyze_graph(self, graph: nx.DiGraph) -> Dict[str, any]:
        """
        Analyze graph properties and return statistics.
        
        Args:
            graph: NetworkX DiGraph to analyze
            
        Returns:
            Dictionary with graph analysis results
        """
        try:
            analysis = {
                'nodes': graph.number_of_nodes(),
                'edges': graph.number_of_edges(),
                'is_connected': nx.is_weakly_connected(graph),
                'number_of_components': nx.number_weakly_connected_components(graph),
                'density': nx.density(graph),
                'is_dag': nx.is_directed_acyclic_graph(graph)
            }
            
            # Calculate degree statistics
            in_degrees = dict(graph.in_degree())
            out_degrees = dict(graph.out_degree())
            
            if in_degrees:
                analysis['avg_in_degree'] = sum(in_degrees.values()) / len(in_degrees)
                analysis['max_in_degree'] = max(in_degrees.values())
                analysis['min_in_degree'] = min(in_degrees.values())
            
            if out_degrees:
                analysis['avg_out_degree'] = sum(out_degrees.values()) / len(out_degrees)
                analysis['max_out_degree'] = max(out_degrees.values())
                analysis['min_out_degree'] = min(out_degrees.values())
            
            # Find nodes with highest in-degree (most linked to)
            if in_degrees:
                max_in_degree = max(in_degrees.values())
                analysis['most_linked_pages'] = [
                    url for url, degree in in_degrees.items() 
                    if degree == max_in_degree
                ]
            
            # Find nodes with highest out-degree (most outgoing links)
            if out_degrees:
                max_out_degree = max(out_degrees.values())
                analysis['most_linking_pages'] = [
                    url for url, degree in out_degrees.items() 
                    if degree == max_out_degree
                ]
            
            return analysis
            
        except Exception as e:
            logging.error(f"Error analyzing graph: {e}")
            return {'error': str(e)}
    
    def get_strongly_connected_components(self, graph: nx.DiGraph) -> List[Set[str]]:
        """
        Get strongly connected components in the graph.
        
        Args:
            graph: NetworkX DiGraph
            
        Returns:
            List of sets, each containing URLs in a strongly connected component
        """
        return list(nx.strongly_connected_components(graph))
    
    def get_weakly_connected_components(self, graph: nx.DiGraph) -> List[Set[str]]:
        """
        Get weakly connected components in the graph.
        
        Args:
            graph: NetworkX DiGraph
            
        Returns:
            List of sets, each containing URLs in a weakly connected component
        """
        return list(nx.weakly_connected_components(graph))
    
    def find_shortest_path(self, graph: nx.DiGraph, source: str, target: str) -> Optional[List[str]]:
        """
        Find shortest path between two nodes.
        
        Args:
            graph: NetworkX DiGraph
            source: Source URL
            target: Target URL
            
        Returns:
            List of URLs forming the shortest path, or None if no path exists
        """
        try:
            return nx.shortest_path(graph, source, target)
        except nx.NetworkXNoPath:
            return None
        except nx.NodeNotFound:
            return None
    
    def get_page_rank(self, graph: nx.DiGraph, alpha: float = 0.85) -> Dict[str, float]:
        """
        Calculate PageRank for all nodes in the graph.
        
        Args:
            graph: NetworkX DiGraph
            alpha: Damping parameter (default: 0.85)
            
        Returns:
            Dictionary mapping URLs to PageRank scores
        """
        try:
            return nx.pagerank(graph, alpha=alpha)
        except Exception as e:
            logging.error(f"Error calculating PageRank: {e}")
            return {}
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get graph generation statistics.
        
        Returns:
            Dictionary with generation statistics
        """
        return self.graph_stats.copy()
    
    def filter_graph_by_degree(self, graph: nx.DiGraph, min_in_degree: int = 0, 
                              min_out_degree: int = 0) -> nx.DiGraph:
        """
        Filter graph by minimum in-degree and out-degree.
        
        Args:
            graph: NetworkX DiGraph
            min_in_degree: Minimum in-degree for nodes to keep
            min_out_degree: Minimum out-degree for nodes to keep
            
        Returns:
            Filtered NetworkX DiGraph
        """
        nodes_to_keep = []
        
        for node in graph.nodes():
            in_degree = graph.in_degree(node)
            out_degree = graph.out_degree(node)
            
            if in_degree >= min_in_degree and out_degree >= min_out_degree:
                nodes_to_keep.append(node)
        
        return graph.subgraph(nodes_to_keep).copy()
    
    def export_graph_data(self, graph: nx.DiGraph) -> Dict[str, any]:
        """
        Export graph data in a format suitable for external analysis.
        
        Args:
            graph: NetworkX DiGraph
            
        Returns:
            Dictionary containing nodes, edges, and metadata
        """
        nodes = []
        for node in graph.nodes():
            nodes.append({
                'id': node,
                'in_degree': graph.in_degree(node),
                'out_degree': graph.out_degree(node)
            })
        
        edges = []
        for source, target, data in graph.edges(data=True):
            edges.append({
                'source': source,
                'target': target,
                'weight': data.get('weight', 1)
            })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'node_count': graph.number_of_nodes(),
                'edge_count': graph.number_of_edges(),
                'is_dag': nx.is_directed_acyclic_graph(graph),
                'is_connected': nx.is_weakly_connected(graph)
            }
        }