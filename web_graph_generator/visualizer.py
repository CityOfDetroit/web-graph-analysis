"""
Graph visualization utilities for web link analysis.

This module provides functionality to create interactive HTML visualizations of web link graphs.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import networkx as nx

# Import plotly for interactive visualizations
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    raise ImportError("Plotly is required for visualizations. Install with: pip install plotly")


class GraphVisualizer:
    """Handles interactive graph visualization using Plotly."""
    
    def __init__(self, graph: nx.DiGraph, base_url: str, metadata: Optional[Dict] = None):
        """
        Initialize graph visualizer.
        
        Args:
            graph: NetworkX DiGraph to visualize
            base_url: Base URL used for scraping
            metadata: Optional metadata about scraping parameters
        """
        self.graph = graph
        self.base_url = base_url
        self.metadata = metadata or {}
        self.title = f"Link Graph for {base_url}"
        
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
                           show_labels: bool = None) -> None:
        """
        Create and save interactive HTML graph visualization.
        
        Args:
            output_path: Path to save the HTML file
            layout: Layout algorithm ('spring', 'circular', 'shell', 'kamada_kawai')
            color_scheme: Color scheme ('default', 'degree', 'pagerank', 'depth')
            node_size_metric: Node sizing metric ('constant', 'degree', 'pagerank')
            show_labels: Whether to show node labels (auto-detect if None)
            
        Raises:
            ValueError: If layout or color scheme is invalid
            IOError: If visualization cannot be saved
        """
        # Validate inputs
        valid_layouts = ['spring', 'circular', 'shell', 'kamada_kawai', 'random']
        if layout not in valid_layouts:
            raise ValueError(f"Invalid layout: {layout}. Valid options: {valid_layouts}")
        
        if color_scheme not in self.color_schemes:
            raise ValueError(f"Invalid color scheme: {color_scheme}. Valid options: {list(self.color_schemes.keys())}")
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Generate layout
        pos = self._generate_layout(layout)
        
        # Auto-detect label display
        if show_labels is None:
            show_labels = self.graph.number_of_nodes() <= 50
        
        # Get node positions
        node_x = [pos[node][0] for node in self.graph.nodes()]
        node_y = [pos[node][1] for node in self.graph.nodes()]
        
        # Get edge positions
        edge_x = []
        edge_y = []
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        )
        
        # Get node properties
        node_colors = self._get_node_colors(color_scheme)
        node_sizes = self._get_node_sizes(node_size_metric)
        
        # Create hover text with full URL and metrics
        hover_text = []
        for node in self.graph.nodes():
            in_degree = self.graph.in_degree(node)
            out_degree = self.graph.out_degree(node)
            hover_info = f"<b>{node}</b><br>"
            hover_info += f"In-degree: {in_degree}<br>"
            hover_info += f"Out-degree: {out_degree}<br>"
            hover_info += f"<i>Click to visit page</i>"
            hover_text.append(hover_info)
        
        # Create node labels
        node_labels = []
        if show_labels:
            for node in self.graph.nodes():
                abbreviated_url = self._abbreviate_url(node)
                node_labels.append(abbreviated_url)
        else:
            node_labels = [''] * len(list(self.graph.nodes()))
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text' if show_labels else 'markers',
            hoverinfo='text',
            hovertext=hover_text,
            text=node_labels,
            textposition="middle center",
            textfont=dict(size=10, color='black'),
            customdata=list(self.graph.nodes()),  # Store URLs for click events
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale='Viridis' if color_scheme == 'pagerank' else 'YlOrRd',
                showscale=color_scheme in ['degree', 'pagerank'],
                line=dict(width=2, color='black')
            ),
            showlegend=False
        )
        
        # Create metadata annotation
        metadata_text = self._create_metadata_text()
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace])
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=self.title,
                x=0.5,
                font=dict(size=16)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=80, l=5, r=5, t=40),  # Increased bottom margin for metadata
            annotations=[
                dict(
                    text="Click nodes to visit pages. Hover for details.",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=-0.02,
                    xanchor="center", yanchor="top",
                    font=dict(color="#888", size=12)
                ),
                dict(
                    text=metadata_text,
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=0.005,
                    xanchor="left", yanchor="bottom",
                    font=dict(color="#666", size=10),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="#ccc",
                    borderwidth=1
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        # Add JavaScript for click events (opens URLs in new tab)
        click_script = """
        <script>
        document.addEventListener('DOMContentLoaded', function() {
            var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
            plotDiv.on('plotly_click', function(data) {
                var point = data.points[0];
                if (point.customdata) {
                    window.open(point.customdata, '_blank');
                }
            });
        });
        </script>
        """
        
        # Save to HTML with click functionality
        html_content = fig.to_html(include_plotlyjs=True, config={'displayModeBar': True})
        
        # Inject click script
        html_content = html_content.replace('</body>', f'{click_script}</body>')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logging.info(f"Interactive graph visualization saved to {output_path}")
    
    def _create_metadata_text(self) -> str:
        """
        Create metadata text for display in the visualization.
        
        Returns:
            Formatted metadata text string
        """
        lines = []
        
        # Add search depth
        max_depth = self.metadata.get('max_depth', 'None')
        lines.append(f"<b>Search Depth</b>:<br> {max_depth}")
        
        # Add URL regex patterns info
        url_patterns = self.metadata.get('url_patterns', 'None')
        lines.append(f"<b>URL Filters</b>:<br> {url_patterns}")
        
        # Add CSS selectors info
        css_selectors = self.metadata.get('css_selectors', 'None')
        lines.append(f"<b>CSS Filters</b>:<br> {css_selectors}")

        # Add graph stats
        lines.append(f"<b>Nodes:</b> {self.graph.number_of_nodes()}, <b>Edges:</b> {self.graph.number_of_edges()}")
        
        return "<br>".join(lines)
    
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
    
    def _get_node_colors(self, color_scheme: str) -> List[float]:
        """
        Get node colors for Plotly visualization.
        
        Args:
            color_scheme: Color scheme name
            
        Returns:
            List of color values for nodes
        """
        if color_scheme == 'default':
            return [0.5] * self.graph.number_of_nodes()
        
        elif color_scheme == 'degree':
            degrees = dict(self.graph.degree())
            max_degree = max(degrees.values()) if degrees else 1
            return [degrees[node] / max_degree for node in self.graph.nodes()]
        
        elif color_scheme == 'pagerank':
            try:
                pagerank = nx.pagerank(self.graph)
                max_pr = max(pagerank.values()) if pagerank else 1
                return [pagerank[node] / max_pr for node in self.graph.nodes()]
            except:
                return [0.5] * self.graph.number_of_nodes()
        
        elif color_scheme == 'depth':
            try:
                centrality = nx.betweenness_centrality(self.graph)
                central_node = max(centrality.keys(), key=lambda x: centrality[x])
                distances = nx.single_source_shortest_path_length(self.graph.to_undirected(), central_node)
                max_distance = max(distances.values()) if distances else 1
                return [1 - (distances.get(node, max_distance) / max_distance) for node in self.graph.nodes()]
            except:
                return [0.5] * self.graph.number_of_nodes()
        
        else:
            return [0.5] * self.graph.number_of_nodes()
    
    def _get_node_sizes(self, size_metric: str) -> List[float]:
        """
        Get node sizes for Plotly visualization.
        
        Args:
            size_metric: Sizing metric name
            
        Returns:
            List of sizes for nodes
        """
        base_size = 20
        max_size = 50
        
        if size_metric == 'constant':
            return [base_size] * self.graph.number_of_nodes()
        
        elif size_metric == 'degree':
            degrees = dict(self.graph.degree())
            max_degree = max(degrees.values()) if degrees else 1
            return [base_size + (degrees[node] / max_degree) * (max_size - base_size) for node in self.graph.nodes()]
        
        elif size_metric == 'pagerank':
            try:
                pagerank = nx.pagerank(self.graph)
                max_pr = max(pagerank.values()) if pagerank else 1
                return [base_size + (pagerank[node] / max_pr) * (max_size - base_size) for node in self.graph.nodes()]
            except:
                return [base_size] * self.graph.number_of_nodes()
        
        else:
            return [base_size] * self.graph.number_of_nodes()
    
    def _abbreviate_url(self, url: str, max_length: int = 25) -> str:
        """
        Abbreviate URL for display in graph nodes.
        
        Args:
            url: Full URL to abbreviate
            max_length: Maximum length of abbreviated URL
            
        Returns:
            Abbreviated URL string
        """
        try:
            parsed = urlparse(url)
            path = parsed.path.rstrip('/')
            
            if not path or path == '/':
                return '/'
            
            # Split path into segments
            segments = [seg for seg in path.split('/') if seg]
            
            if not segments:
                return '/'
            
            # If path is short enough, return as-is
            full_path = '/' + '/'.join(segments)
            if len(full_path) <= max_length:
                return full_path
            
            # If only one segment, truncate it
            if len(segments) == 1:
                segment = segments[0]
                if len(segment) <= max_length - 1:
                    return '/' + segment
                else:
                    return '/' + segment[:max_length-4] + '...'
            
            # Multiple segments - show first and last with ... in between
            first_segment = segments[0]
            last_segment = segments[-1]
            
            # Try different combinations to fit within max_length
            abbreviated = f"/{first_segment}/.../{last_segment}"
            
            if len(abbreviated) <= max_length:
                return abbreviated
            
            # If still too long, truncate the segments
            available_length = max_length - 5  # Account for "/.../"
            first_max = available_length // 2
            last_max = available_length - first_max
            
            first_truncated = first_segment[:first_max] + ('...' if len(first_segment) > first_max else '')
            last_truncated = last_segment[:last_max] + ('...' if len(last_segment) > last_max else '')
            
            return f"/{first_truncated}/.../{last_truncated}"
            
        except Exception:
            # Fallback to simple truncation
            return url[:max_length-3] + '...' if len(url) > max_length else url
    
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
            self.base_url,
            self.metadata
        )
        sub_visualizer.title = f"Subgraph for {self.base_url} ({subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges)"
        
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
                output_path = output_dir / f"component_{i+1}.html"
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