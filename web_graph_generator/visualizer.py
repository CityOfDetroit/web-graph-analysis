"""
Graph visualization utilities for web link analysis.

This module provides functionality to create visual representations of web link graphs.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


class GraphVisualizer:
    """Handles graph visualization and image generation."""
    
    def __init__(self, graph: nx.DiGraph, title: Optional[str] = None):
        """
        Initialize graph visualizer.
        
        Args:
            graph: NetworkX DiGraph to visualize
            title: Optional title for the visualization
        """
        self.graph = graph
        self.title = title or f"Web Link Graph ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)"
        
        # Color schemes
        self.color_schemes = {
            'default': {'node': 'lightblue', 'edge': 'gray'},
            'degree': {'node': 'degree_based', 'edge': 'gray'},
            'pagerank': {'node': 'pagerank_based', 'edge': 'gray'},
            'depth': {'node': 'depth_based', 'edge': 'gray'}
        }
    
    def create_visualization(self, output_path: str, layout: str = 'spring',
                           color_scheme: str = 'default', 
                           node_size_metric: str = 'constant',
                           show_labels: bool = None,
                           figsize: Tuple[int, int] = (16, 12),
                           dpi: int = 300) -> None:
        """
        Create and save graph visualization.
        
        Args:
            output_path: Path to save the SVG image
            layout: Layout algorithm ('spring', 'circular', 'shell', 'kamada_kawai')
            color_scheme: Color scheme ('default', 'degree', 'pagerank', 'depth')
            node_size_metric: Node sizing metric ('constant', 'degree', 'pagerank')
            show_labels: Whether to show node labels (auto-detect if None)
            figsize: Figure size in inches
            dpi: Resolution for raster formats
            
        Raises:
            ValueError: If layout or color scheme is invalid
            IOError: If visualization cannot be saved
        """
        try:
            # Validate inputs
            valid_layouts = ['spring', 'circular', 'shell', 'kamada_kawai', 'random']
            if layout not in valid_layouts:
                raise ValueError(f"Invalid layout: {layout}. Valid options: {valid_layouts}")
            
            if color_scheme not in self.color_schemes:
                raise ValueError(f"Invalid color scheme: {color_scheme}. Valid options: {list(self.color_schemes.keys())}")
            
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Create figure
            plt.figure(figsize=figsize, dpi=dpi)
            
            # Generate layout
            pos = self._generate_layout(layout)
            
            # Determine node colors and sizes
            node_colors = self._get_node_colors(color_scheme)
            node_sizes = self._get_node_sizes(node_size_metric)
            
            # Determine edge properties
            edge_widths = self._get_edge_widths()
            edge_colors = self._get_edge_colors()
            
            # Auto-detect label display
            if show_labels is None:
                show_labels = self.graph.number_of_nodes() <= 25
            
            # Draw nodes
            nx.draw_networkx_nodes(
                self.graph, pos,
                node_color=node_colors,
                node_size=node_sizes,
                alpha=0.8,
                linewidths=1,
                edgecolors='black'
            )
            
            # Draw edges
            nx.draw_networkx_edges(
                self.graph, pos,
                width=edge_widths,
                alpha=0.6,
                arrows=True,
                arrowsize=20,
                arrowstyle='->',
                edge_color=edge_colors,
                connectionstyle='arc3,rad=0.1'
            )
            
            # Draw labels if requested
            if show_labels:
                labels = self._generate_labels()
                nx.draw_networkx_labels(
                    self.graph, pos,
                    labels,
                    font_size=8,
                    font_weight='bold',
                    font_color='black'
                )
            
            # Add title and metadata
            plt.title(self.title, fontsize=16, fontweight='bold', pad=20)
            
            # Add legend if using special color schemes
            if color_scheme in ['degree', 'pagerank']:
                self._add_colorbar(color_scheme)
            
            # Remove axes
            plt.axis('off')
            plt.tight_layout()
            
            # Save the visualization
            file_format = Path(output_path).suffix.lower().lstrip('.')
            if file_format == 'svg':
                plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=dpi)
            elif file_format in ['png', 'jpg', 'jpeg', 'pdf']:
                plt.savefig(output_path, format=file_format, bbox_inches='tight', dpi=dpi)
            else:
                # Default to SVG
                plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=dpi)
            
            plt.close()
            
            logging.info(f"Graph visualization saved to {output_path}")
            
        except Exception as e:
            logging.error(f"Failed to create visualization: {e}")
            raise IOError(f"Could not create visualization: {e}")
    
    def _generate_layout(self, layout: str) -> Dict[str, Tuple[float, float]]:
        """
        Generate node positions based on layout algorithm.
        
        Args:
            layout: Layout algorithm name
            
        Returns:
            Dictionary mapping nodes to (x, y) positions
        """
        if layout == 'spring':
            if self.graph.number_of_nodes() < 50:
                return nx.spring_layout(self.graph, k=3, iterations=50, seed=42)
            else:
                return nx.spring_layout(self.graph, k=1, iterations=20, seed=42)
        elif layout == 'circular':
            return nx.circular_layout(self.graph)
        elif layout == 'shell':
            return nx.shell_layout(self.graph)
        elif layout == 'kamada_kawai':
            return nx.kamada_kawai_layout(self.graph)
        elif layout == 'random':
            return nx.random_layout(self.graph, seed=42)
        else:
            return nx.spring_layout(self.graph, seed=42)
    
    def _get_node_colors(self, color_scheme: str) -> List[str]:
        """
        Get node colors based on color scheme.
        
        Args:
            color_scheme: Color scheme name
            
        Returns:
            List of colors for nodes
        """
        if color_scheme == 'default':
            return ['lightblue'] * self.graph.number_of_nodes()
        
        elif color_scheme == 'degree':
            degrees = dict(self.graph.degree())
            max_degree = max(degrees.values()) if degrees else 1
            
            # Create color map based on degree
            colors = []
            for node in self.graph.nodes():
                degree_ratio = degrees[node] / max_degree
                # Use a color gradient from light blue to dark red
                colors.append(plt.cm.YlOrRd(degree_ratio))
            return colors
        
        elif color_scheme == 'pagerank':
            try:
                pagerank = nx.pagerank(self.graph)
                max_pr = max(pagerank.values()) if pagerank else 1
                
                colors = []
                for node in self.graph.nodes():
                    pr_ratio = pagerank[node] / max_pr
                    colors.append(plt.cm.viridis(pr_ratio))
                return colors
            except:
                # Fallback to default if PageRank fails
                return ['lightblue'] * self.graph.number_of_nodes()
        
        elif color_scheme == 'depth':
            # Color nodes by their distance from the most central node
            try:
                # Find the node with highest betweenness centrality
                centrality = nx.betweenness_centrality(self.graph)
                central_node = max(centrality.keys(), key=lambda x: centrality[x])
                
                # Calculate distances from central node
                distances = nx.single_source_shortest_path_length(self.graph.to_undirected(), central_node)
                max_distance = max(distances.values()) if distances else 1
                
                colors = []
                for node in self.graph.nodes():
                    distance = distances.get(node, max_distance)
                    depth_ratio = distance / max_distance
                    colors.append(plt.cm.plasma(1 - depth_ratio))  # Invert so central is bright
                return colors
            except:
                return ['lightblue'] * self.graph.number_of_nodes()
        
        else:
            return ['lightblue'] * self.graph.number_of_nodes()
    
    def _get_node_sizes(self, size_metric: str) -> List[float]:
        """
        Get node sizes based on metric.
        
        Args:
            size_metric: Sizing metric name
            
        Returns:
            List of sizes for nodes
        """
        base_size = 300
        
        if size_metric == 'constant':
            return [base_size] * self.graph.number_of_nodes()
        
        elif size_metric == 'degree':
            degrees = dict(self.graph.degree())
            max_degree = max(degrees.values()) if degrees else 1
            
            sizes = []
            for node in self.graph.nodes():
                degree_ratio = degrees[node] / max_degree
                size = base_size + (degree_ratio * base_size * 2)
                sizes.append(size)
            return sizes
        
        elif size_metric == 'pagerank':
            try:
                pagerank = nx.pagerank(self.graph)
                max_pr = max(pagerank.values()) if pagerank else 1
                
                sizes = []
                for node in self.graph.nodes():
                    pr_ratio = pagerank[node] / max_pr
                    size = base_size + (pr_ratio * base_size * 2)
                    sizes.append(size)
                return sizes
            except:
                return [base_size] * self.graph.number_of_nodes()
        
        else:
            return [base_size] * self.graph.number_of_nodes()
    
    def _get_edge_widths(self) -> List[float]:
        """
        Get edge widths based on weights.
        
        Returns:
            List of widths for edges
        """
        edges = self.graph.edges(data=True)
        weights = [edge[2].get('weight', 1) for edge in edges]
        
        if not weights:
            return [1.0]
        
        max_weight = max(weights)
        min_width = 0.5
        max_width = 4.0
        
        widths = []
        for weight in weights:
            normalized_weight = weight / max_weight
            width = min_width + (normalized_weight * (max_width - min_width))
            widths.append(width)
        
        return widths
    
    def _get_edge_colors(self) -> List[str]:
        """
        Get edge colors based on weights.
        
        Returns:
            List of colors for edges
        """
        edges = self.graph.edges(data=True)
        weights = [edge[2].get('weight', 1) for edge in edges]
        
        if not weights:
            return ['gray']
        
        max_weight = max(weights)
        colors = []
        
        for weight in weights:
            intensity = weight / max_weight
            # Use grayscale: lighter for lower weights, darker for higher weights
            gray_value = 0.8 - (intensity * 0.6)  # Range from 0.8 to 0.2
            colors.append(str(gray_value))
        
        return colors
    
    def _generate_labels(self) -> Dict[str, str]:
        """
        Generate node labels from URLs.
        
        Returns:
            Dictionary mapping nodes to display labels
        """
        labels = {}
        for node in self.graph.nodes():
            # Extract meaningful part of URL for display
            if '/' in node:
                parts = node.split('/')
                if len(parts) > 3:
                    # Take the last non-empty part, or the domain if path is empty
                    label = parts[-1] if parts[-1] else parts[-2]
                else:
                    label = parts[-1] if parts[-1] else node
            else:
                label = node
            
            # Limit label length
            if len(label) > 15:
                label = label[:12] + '...'
            
            labels[node] = label
        
        return labels
    
    def _add_colorbar(self, color_scheme: str) -> None:
        """
        Add a colorbar legend to the plot.
        
        Args:
            color_scheme: Color scheme used
        """
        # This is a simplified colorbar - in a real implementation,
        # you would create a proper colorbar with the actual values
        pass
    
    def create_subgraph_visualization(self, nodes: List[str], output_path: str, 
                                    **kwargs) -> None:
        """
        Create visualization of a subgraph.
        
        Args:
            nodes: List of nodes to include in subgraph
            output_path: Path to save the visualization
            **kwargs: Additional arguments for create_visualization
        """
        if not nodes:
            raise ValueError("Cannot create subgraph with empty node list")
        
        # Create subgraph
        subgraph = self.graph.subgraph(nodes).copy()
        
        # Create new visualizer for subgraph
        sub_visualizer = GraphVisualizer(
            subgraph, 
            title=f"Subgraph ({subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges)"
        )
        
        # Create visualization
        sub_visualizer.create_visualization(output_path, **kwargs)
    
    def create_component_visualizations(self, output_dir: str, 
                                      component_type: str = 'weakly_connected',
                                      **kwargs) -> List[str]:
        """
        Create separate visualizations for each connected component.
        
        Args:
            output_dir: Directory to save component visualizations
            component_type: Type of components ('weakly_connected' or 'strongly_connected')
            **kwargs: Additional arguments for create_visualization
            
        Returns:
            List of paths to created visualizations
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if component_type == 'weakly_connected':
            components = list(nx.weakly_connected_components(self.graph))
        elif component_type == 'strongly_connected':
            components = list(nx.strongly_connected_components(self.graph))
        else:
            raise ValueError(f"Invalid component type: {component_type}")
        
        created_files = []
        
        for i, component in enumerate(components):
            if len(component) > 1:  # Skip single-node components
                output_path = output_dir / f"component_{i+1}.svg"
                self.create_subgraph_visualization(
                    list(component), 
                    str(output_path), 
                    **kwargs
                )
                created_files.append(str(output_path))
        
        logging.info(f"Created {len(created_files)} component visualizations in {output_dir}")
        return created_files
    
    def get_visualization_stats(self) -> Dict[str, any]:
        """
        Get statistics about the graph for visualization purposes.
        
        Returns:
            Dictionary with visualization-relevant statistics
        """
        stats = {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph),
            'components': nx.number_weakly_connected_components(self.graph)
        }
        
        # Degree statistics
        degrees = dict(self.graph.degree())
        if degrees:
            stats['avg_degree'] = sum(degrees.values()) / len(degrees)
            stats['max_degree'] = max(degrees.values())
            stats['min_degree'] = min(degrees.values())
        
        # Recommended visualization settings
        if stats['nodes'] <= 10:
            stats['recommended_layout'] = 'circular'
            stats['recommended_labels'] = True
        elif stats['nodes'] <= 50:
            stats['recommended_layout'] = 'spring'
            stats['recommended_labels'] = True
        else:
            stats['recommended_layout'] = 'spring'
            stats['recommended_labels'] = False
        
        return stats