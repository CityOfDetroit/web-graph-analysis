#!/usr/bin/env python3
"""
Web Page Link Graph Generator - Main Entry Point

A script that generates interactive HTML graphs of web pages and their links
or Drupal taxonomy hierarchies. Supports web scraping or loading from pre-existing data files.
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
    from .drupal_taxonomy import DrupalTaxonomyExtractor, test_connection
except ImportError:
    # Fallback to absolute imports (when run directly)
    from web_graph_generator.url_handler import URLFilter
    from web_graph_generator.scraper import WebScraper, CSSFilter
    from web_graph_generator.graph_generator import GraphGenerator
    from web_graph_generator.data_serializer import DataSerializer
    from web_graph_generator.visualizer import GraphVisualizer
    from web_graph_generator.drupal_taxonomy import DrupalTaxonomyExtractor, test_connection


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
    # Check data source requirements
    if args.data_source == 'web-scraper':
        if not args.scrape and not args.data_file:
            logging.error("For web-scraper: either --scrape must be enabled or --data-file must be provided")
            sys.exit(1)
        
        if args.scrape and args.data_file:
            logging.warning("Both --scrape and --data-file provided. Using scraping mode.")
        
        # Validate URLs for web scraping
        if args.scrape:
            if not args.base_url or not args.base_url.startswith(('http://', 'https://')):
                logging.error("Base URL must start with http:// or https:// for web scraping")
                sys.exit(1)
    
    elif args.data_source == 'drupal-taxonomy':
        if not args.base_term_id:
            logging.error("For drupal-taxonomy: --base-term-id is required")
            sys.exit(1)
        
        if args.base_term_id <= 0:
            logging.error("Base term ID must be a positive integer")
            sys.exit(1)
    
    else:
        logging.error(f"Unknown data source: {args.data_source}")
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
        description='Generate interactive HTML graphs of web pages and links or Drupal taxonomy hierarchies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape a website and save data + generate interactive graph
  python -m web_graph_generator --data-source web-scraper --base-url https://example.com --scrape --max-depth 3 --output-data graph_data.json --data-format json --output-image graph.html

  # Load existing web scraping data and generate interactive graph
  python -m web_graph_generator --data-source web-scraper --base-url https://example.com --data-file graph_data.json --output-image graph.html

  # Extract Drupal taxonomy hierarchy (Lando defaults)
  python -m web_graph_generator --data-source drupal-taxonomy --base-term-id 123 --base-url https://example.com --max-depth 4 --output-image taxonomy.html

  # Extract taxonomy with custom database settings
  python -m web_graph_generator --data-source drupal-taxonomy --base-term-id 45 --base-url https://detroitmi.gov --db-host localhost --db-name my_drupal --db-user myuser --db-password mypass --output-image departments.html

  # Test database connection
  python -m web_graph_generator --data-source drupal-taxonomy --test-db-connection --db-host database --db-name drupal10
        """
    )
    
    # Data source selection
    parser.add_argument(
        '--data-source',
        choices=['web-scraper', 'drupal-taxonomy'],
        default='web-scraper',
        help='Data source type (default: web-scraper)'
    )
    
    # Common arguments
    parser.add_argument(
        '--base-url',
        help='Base URL for the site (required for web-scraper, optional for drupal-taxonomy for URL construction)'
    )
    
    parser.add_argument(
        '--max-depth',
        type=int,
        default=2,
        help='Maximum depth to traverse (default: 2)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # Web scraper arguments
    web_group = parser.add_argument_group('Web Scraper Options')
    web_group.add_argument(
        '--scrape',
        action='store_true',
        help='Enable web scraping mode (for web-scraper data source)'
    )
    
    web_group.add_argument(
        '--data-file',
        help='Path to pre-existing serialized data file (for web-scraper when not scraping)'
    )
    
    web_group.add_argument(
        '--skip-patterns',
        help='File containing regex patterns of URLs to skip'
    )
    
    web_group.add_argument(
        '--skip-selectors',
        help='File containing CSS selectors for HTML elements to skip when extracting links'
    )
    
    web_group.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Delay between requests in seconds (default: 1.0)'
    )
    
    web_group.add_argument(
        '--timeout',
        type=int,
        default=10,
        help='Request timeout in seconds (default: 10)'
    )
    
    # Drupal taxonomy arguments
    drupal_group = parser.add_argument_group('Drupal Taxonomy Options')
    drupal_group.add_argument(
        '--base-term-id',
        type=int,
        help='Starting taxonomy term ID (required for drupal-taxonomy)'
    )
    
    drupal_group.add_argument(
        '--db-host',
        default='database',
        help='Database host (default: database - for Lando)'
    )
    
    drupal_group.add_argument(
        '--db-port',
        type=int,
        default=3306,
        help='Database port (default: 3306)'
    )
    
    drupal_group.add_argument(
        '--db-name',
        default='drupal10',
        help='Database name (default: drupal10)'
    )
    
    drupal_group.add_argument(
        '--db-user',
        default='drupal10',
        help='Database user (default: drupal10)'
    )
    
    drupal_group.add_argument(
        '--db-password',
        default='drupal10',
        help='Database password (default: drupal10)'
    )
    
    drupal_group.add_argument(
        '--test-db-connection',
        action='store_true',
        help='Test database connection and exit'
    )
    
    # Graph options
    graph_group = parser.add_argument_group('Graph Options')
    graph_group.add_argument(
        '--allow-cycles',
        action='store_true',
        default=True,
        help='Allow cycles in the graph (default: True)'
    )
    
    graph_group.add_argument(
        '--no-cycles',
        action='store_true',
        help='Disable cycles in the graph'
    )
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--output-data',
        help='Output path for serialized data'
    )
    
    output_group.add_argument(
        '--data-format',
        choices=['pickle', 'json'],
        default='pickle',
        help='Format for serialized data output (default: pickle)'
    )
    
    output_group.add_argument(
        '--output-image',
        default='graph.html',
        help='Output path for interactive HTML graph (default: graph.html)'
    )
    
    # Visualization options
    viz_group = parser.add_argument_group('Visualization Options')
    viz_group.add_argument(
        '--layout',
        choices=['spring', 'circular', 'shell', 'kamada_kawai', 'random', 'hierarchical'],
        default='spring',
        help='Graph layout algorithm (default: spring)'
    )
    
    viz_group.add_argument(
        '--color-scheme',
        choices=['default', 'degree', 'pagerank', 'depth'],
        default='default',
        help='Node color scheme (default: default)'
    )
    
    viz_group.add_argument(
        '--node-size',
        choices=['constant', 'degree', 'pagerank'],
        default='constant',
        help='Node size metric (default: constant)'
    )
    
    viz_group.add_argument(
        '--show-labels',
        action='store_true',
        help='Force show node labels'
    )
    
    viz_group.add_argument(
        '--hide-labels',
        action='store_true',
        help='Force hide node labels'
    )
    
    return parser


def extract_drupal_taxonomy(args: argparse.Namespace) -> tuple[dict, dict]:
    """
    Extract Drupal taxonomy hierarchy and return graph data and metadata.
    
    Args:
        args: Command line arguments
        
    Returns:
        Tuple of (graph_data, metadata)
    """
    logging.info("Starting Drupal taxonomy extraction mode")
    
    # Build database configuration
    db_config = {
        'host': args.db_host,
        'port': args.db_port,
        'database': args.db_name,
        'user': args.db_user,
        'password': args.db_password,
        'charset': 'utf8mb4',
        'use_unicode': True,
        'autocommit': True
    }
    
    # Test connection if requested
    if args.test_db_connection:
        success = test_connection(db_config)
        if success:
            logging.info("Database connection test successful!")
            sys.exit(0)
        else:
            logging.error("Database connection test failed!")
            sys.exit(1)
    
    # Create taxonomy extractor
    extractor = DrupalTaxonomyExtractor(
        db_config=db_config,
        base_term_id=args.base_term_id,
        max_depth=args.max_depth,
        base_url=args.base_url
    )
    
    try:
        # Connect and validate base term
        extractor.connect()
        
        if not extractor.validate_term_exists(args.base_term_id):
            logging.error(f"Base term ID {args.base_term_id} does not exist in database")
            sys.exit(1)
        
        # Get term info for logging
        term_info = extractor.get_term_info(args.base_term_id)
        if term_info:
            logging.info(f"Starting from term: {term_info['name']} (ID: {term_info['tid']}, Vocabulary: {term_info['vocabulary_name']})")
        
        # Extract taxonomy hierarchy
        graph_data = extractor.get_taxonomy_hierarchy()
        
        # Log extraction statistics
        stats = extractor.get_statistics()
        logging.info(f"Extraction statistics: {stats}")
        
        # Collect metadata for visualization
        metadata = {
            'data_source': 'drupal-taxonomy',
            'base_term_id': args.base_term_id,
            'base_term_info': term_info,
            'max_depth': args.max_depth,
            'database': args.db_name,
            'extraction_stats': stats
        }
        
        # Save data if requested
        if args.output_data:
            DataSerializer.save_graph_data(graph_data, args.output_data, args.data_format)
        
        return graph_data, metadata
        
    finally:
        # Clean up database connection
        extractor.close()


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
            'data_source': 'web-scraper',
            'max_depth': args.max_depth + 1,
            'url_patterns': url_filter.get_patterns_html() if url_filter else 'None',
            'css_selectors': css_filter.get_selectors_html() if css_filter else 'None',
            'allow_cycles': allow_cycles,
            'scraping_stats': stats
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
        'data_source': 'web-scraper-file',
        'data_file': args.data_file,
        'max_depth': args.max_depth + 1,
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
        metadata: Metadata about scraping/extraction parameters
    """
    logging.info("Creating visualization")
    
    # Use base URL from args, or extract from metadata if available
    base_url = args.base_url
    if not base_url and args.data_source == 'drupal-taxonomy':
        # For taxonomy without base URL, use a generic identifier
        term_info = metadata.get('base_term_info', {})
        base_url = f"Taxonomy: {term_info.get('name', f'Term {args.base_term_id}')}"
    elif not base_url:
        base_url = "Unknown"
    
    # Create visualizer with metadata
    visualizer = GraphVisualizer(graph, base_url, metadata)
    
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
        # Get graph data and metadata based on data source
        if args.data_source == 'drupal-taxonomy':
            graph_data, metadata = extract_drupal_taxonomy(args)
        elif args.data_source == 'web-scraper':
            if args.scrape:
                graph_data, metadata = scrape_website(args)
            else:
                graph_data, metadata = load_existing_data(args)
        else:
            logging.error(f"Unknown data source: {args.data_source}")
            sys.exit(1)
        
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