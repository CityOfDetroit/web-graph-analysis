#!/usr/bin/env python3
"""
Web Page Link Graph Generator - Main Entry Point

A script that generates interactive HTML graphs of web pages and their links.
Supports web scraping or loading from pre-existing data files.
"""

import argparse
import logging
import sys
from pathlib import Path

try:
    # Try relative imports first (when run as module)
    from .url_handler import URLFilter
    from .scraper import WebScraper, CSSFilter
    from .graph_generator import GraphGenerator
    from .data_serializer import DataSerializer
    from .visualizer import GraphVisualizer
except ImportError:
    # Fallback to absolute imports (when run directly)
    from web_graph_generator.url_handler import URLFilter
    from web_graph_generator.scraper import WebScraper, CSSFilter
    from web_graph_generator.graph_generator import GraphGenerator
    from web_graph_generator.data_serializer import DataSerializer
    from web_graph_generator.visualizer import GraphVisualizer


def setup_logging(verbose: bool) -> None:
    """
    Setup logging configuration.
    
    Args:
        verbose: Whether to enable verbose logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def validate_arguments(args: argparse.Namespace) -> None:
    """
    Validate command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Raises:
        SystemExit: If arguments are invalid
    """
    # Check mutually exclusive scraping options
    if not args.scrape and not args.data_file:
        logging.error("Either --scrape must be enabled or --data-file must be provided")
        sys.exit(1)
    
    if args.scrape and args.data_file:
        logging.warning("Both --scrape and --data-file provided. Using scraping mode.")
    
    # Validate URLs
    if args.scrape:
        if not args.base_url.startswith(('http://', 'https://')):
            logging.error("Base URL must start with http:// or https://")
            sys.exit(1)
    
    # Validate depth
    if args.max_depth < 0:
        logging.error("Maximum depth must be non-negative")
        sys.exit(1)
    
    # Validate files
    if args.skip_patterns and not Path(args.skip_patterns).exists():
        logging.error(f"Skip patterns file not found: {args.skip_patterns}")
        sys.exit(1)
    
    if args.skip_selectors and not Path(args.skip_selectors).exists():
        logging.error(f"CSS selectors file not found: {args.skip_selectors}")
        sys.exit(1)
    
    if args.data_file and not Path(args.data_file).exists():
        logging.error(f"Data file not found: {args.data_file}")
        sys.exit(1)
    
    # Validate output paths
    if args.output_data:
        output_dir = Path(args.output_data).parent
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logging.error(f"Cannot create output directory: {e}")
                sys.exit(1)
    
    if args.output_image:
        output_dir = Path(args.output_image).parent
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logging.error(f"Cannot create output directory: {e}")
                sys.exit(1)


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure argument parser.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description='Generate interactive HTML graphs of web pages and links',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape a website and save data + generate interactive graph
  python -m web_graph_generator --base-url https://example.com --scrape --max-depth 3 --output-data graph_data.json --data-format json --output-image graph.html

  # Load existing data and generate interactive graph
  python -m web_graph_generator --base-url https://example.com --data-file graph_data.json --output-image graph.html

  # Scrape with URL filtering and CSS selector filtering
  python -m web_graph_generator --base-url https://example.com --scrape --skip-patterns skip_urls.txt --skip-selectors skip_elements.txt --output-image graph.html --verbose

  # Scrape with no cycles
  python -m web_graph_generator --base-url https://example.com --scrape --no-cycles --output-image graph.html --verbose
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--base-url',
        required=True,
        help='Base URL to start crawling from'
    )
    
    # Optional arguments
    parser.add_argument(
        '--max-depth',
        type=int,
        default=2,
        help='Maximum depth to traverse from base URL (default: 2)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--skip-patterns',
        help='File containing regex patterns of URLs to skip'
    )
    
    parser.add_argument(
        '--skip-selectors',
        help='File containing CSS selectors for HTML elements to skip when extracting links'
    )
    
    # Scraping options
    parser.add_argument(
        '--scrape',
        action='store_true',
        help='Enable web scraping mode'
    )
    
    parser.add_argument(
        '--data-file',
        help='Path to pre-existing serialized data file (required if --scrape not used)'
    )
    
    # Graph options
    parser.add_argument(
        '--allow-cycles',
        action='store_true',
        default=True,
        help='Allow cycles in the graph (default: True)'
    )
    
    parser.add_argument(
        '--no-cycles',
        action='store_true',
        help='Disable cycles in the graph'
    )
    
    # Output options
    parser.add_argument(
        '--output-data',
        help='Output path for serialized data (only used with --scrape)'
    )
    
    parser.add_argument(
        '--data-format',
        choices=['pickle', 'json'],
        default='pickle',
        help='Format for serialized data output (default: pickle)'
    )
    
    parser.add_argument(
        '--output-image',
        default='graph.html',
        help='Output path for interactive HTML graph (default: graph.html)'
    )
    
    # Visualization options
    parser.add_argument(
        '--layout',
        choices=['spring', 'circular', 'shell', 'kamada_kawai', 'random'],
        default='spring',
        help='Graph layout algorithm (default: spring)'
    )
    
    parser.add_argument(
        '--color-scheme',
        choices=['default', 'degree', 'pagerank', 'depth'],
        default='default',
        help='Node color scheme (default: default)'
    )
    
    parser.add_argument(
        '--node-size',
        choices=['constant', 'degree', 'pagerank'],
        default='constant',
        help='Node size metric (default: constant)'
    )
    
    parser.add_argument(
        '--show-labels',
        action='store_true',
        help='Force show node labels'
    )
    
    parser.add_argument(
        '--hide-labels',
        action='store_true',
        help='Force hide node labels'
    )
    
    # Scraping options
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Delay between requests in seconds (default: 1.0)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=10,
        help='Request timeout in seconds (default: 10)'
    )
    
    return parser


def scrape_website(args: argparse.Namespace) -> tuple[dict, dict]:
    """
    Scrape website and return graph data and metadata.
    
    Args:
        args: Command line arguments
        
    Returns:
        Tuple of (graph_data, metadata)
    """
    logging.info("Starting web scraping mode")
    
    # Initialize URL filter
    url_filter = URLFilter(args.skip_patterns)
    
    # Initialize CSS filter
    css_filter = CSSFilter(args.skip_selectors) if args.skip_selectors else None
    
    # Handle cycle settings
    allow_cycles = args.allow_cycles and not args.no_cycles
    
    # Create scraper
    scraper = WebScraper(
        base_url=args.base_url,
        max_depth=args.max_depth,
        url_filter=url_filter,
        css_filter=css_filter,
        allow_cycles=allow_cycles,
        delay=args.delay,
        timeout=args.timeout
    )
    
    try:
        # Scrape the website
        graph_data = scraper.scrape()
        
        # Log scraping statistics
        stats = scraper.get_statistics()
        logging.info(f"Scraping statistics: {stats}")
        
        # Collect metadata for visualization
        metadata = {
            'max_depth': args.max_depth,
            'url_patterns': url_filter.get_patterns_html() if url_filter else 'None',
            'css_selectors': css_filter.get_selectors_html() if css_filter else 'None',
            'allow_cycles': allow_cycles
        }
        
        # Save data if requested
        if args.output_data:
            DataSerializer.save_graph_data(graph_data, args.output_data, args.data_format)
        
        return graph_data, metadata
        
    finally:
        # Clean up scraper resources
        scraper.close()


def load_existing_data(args: argparse.Namespace) -> tuple[dict, dict]:
    """
    Load existing graph data from file.
    
    Args:
        args: Command line arguments
        
    Returns:
        Tuple of (graph_data, metadata)
    """
    logging.info("Loading data from file")
    
    # Get file info for logging
    file_info = DataSerializer.get_file_info(args.data_file)
    logging.info(f"Data file info: {file_info}")
    
    # Load the data
    graph_data = DataSerializer.load_graph_data(args.data_file)
    
    logging.info(f"Loaded {len(graph_data)} pages from {args.data_file}")

    # Initialize URL filter
    url_filter = URLFilter(args.skip_patterns)
    
    # Initialize CSS filter
    css_filter = CSSFilter(args.skip_selectors) if args.skip_selectors else None
    
    # Handle cycle settings
    allow_cycles = args.allow_cycles and not args.no_cycles
    
    # Collect metadata for visualization
    metadata = {
        'max_depth': args.max_depth,
        'url_patterns': url_filter.get_patterns_html() if url_filter else 'None',
        'css_selectors': css_filter.get_selectors_html() if css_filter else 'None',
        'allow_cycles': allow_cycles
    }
    
    return graph_data, metadata


def create_graph(graph_data: dict, args: argparse.Namespace) -> 'nx.DiGraph':
    """
    Create NetworkX graph from data.
    
    Args:
        graph_data: Graph data dictionary
        args: Command line arguments
        
    Returns:
        NetworkX DiGraph
    """
    logging.info("Creating graph from data")
    
    # Handle cycle settings
    allow_cycles = args.allow_cycles and not args.no_cycles
    
    # Create graph generator
    graph_generator = GraphGenerator(allow_cycles=allow_cycles)
    
    # Generate graph
    graph = graph_generator.create_graph(graph_data)
    
    # Log graph statistics
    stats = graph_generator.get_statistics()
    logging.info(f"Graph generation statistics: {stats}")
    
    # Analyze graph
    analysis = graph_generator.analyze_graph(graph)
    logging.info(f"Graph analysis: {analysis}")
    
    return graph


def create_visualization(graph: 'nx.DiGraph', args: argparse.Namespace, metadata: dict) -> None:
    """
    Create and save graph visualization.
    
    Args:
        graph: NetworkX DiGraph
        args: Command line arguments
        metadata: Metadata about scraping parameters
    """
    logging.info("Creating visualization")
    
    # Create visualizer with metadata
    visualizer = GraphVisualizer(graph, args.base_url, metadata)
    
    # Get visualization recommendations
    viz_stats = visualizer.get_visualization_stats()
    logging.info(f"Visualization stats: {viz_stats}")
    
    # Determine label display
    show_labels = None
    if args.show_labels:
        show_labels = True
    elif args.hide_labels:
        show_labels = False
    
    # Create visualization
    visualizer.create_visualization(
        output_path=args.output_image,
        layout=args.layout,
        color_scheme=args.color_scheme,
        node_size_metric=args.node_size,
        show_labels=show_labels
    )


def main():
    """Main function."""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Validate arguments
    validate_arguments(args)
    
    try:
        # Get graph data and metadata
        if args.scrape:
            graph_data, metadata = scrape_website(args)
        else:
            graph_data, metadata = load_existing_data(args)
        
        # Create graph
        graph = create_graph(graph_data, args)
        
        # Create visualization
        create_visualization(graph, args, metadata)
        
        logging.info("Processing complete!")
        
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error during processing: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()