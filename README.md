# Web Graph Generator

A Python package for generating directed graphs of web pages and their links. This tool can scrape websites or load existing data to create visualizations and analyze link structures.

## Features

- **Web Scraping**: Breadth-first traversal of websites with configurable depth
- **Graph Generation**: Creates weighted directed graphs using NetworkX
- **URL Filtering**: Regex-based URL filtering with customizable patterns
- **Multiple Formats**: Support for pickle and JSON data serialization
- **Rich Visualizations**: SVG output with multiple layout algorithms and color schemes
- **Cycle Handling**: Optional cycle detection and prevention
- **Modular Design**: Clean, extensible architecture with separate modules

## Quick Start

### Setting Up The Environment

```bash
cd web-graph-generator
pip install -e .
```

### Scrape a Website

```bash
python -m web_graph_generator --base-url https://example.com --scrape --max-depth 2 --output-data graph_data.json --data-format json --output-image graph.svg
```

### Load Existing Data

```bash
python -m web_graph_generator --base-url https://example.com --data-file graph_data.json --output-image loaded_graph.svg
```

## Usage

### Command Line Arguments

#### Required Arguments
- `--base-url`: Starting URL for crawling

#### Scraping Options
- `--scrape`: Enable web scraping mode
- `--data-file`: Path to pre-existing data file (required if not scraping)
- `--max-depth`: Maximum crawling depth (default: 2)
- `--delay`: Delay between requests in seconds (default: 1.0)
- `--timeout`: Request timeout in seconds (default: 10)

#### Graph Options
- `--allow-cycles`: Allow cycles in the graph (default: True)
- `--no-cycles`: Disable cycles in the graph
- `--skip-patterns`: File with regex patterns to skip URLs

#### Output Options
- `--output-data`: Path for serialized data output
- `--data-format`: Format for data (pickle or json)
- `--output-image`: Path for graph image (default: graph.svg)

#### Visualization Options
- `--layout`: Layout algorithm (spring, circular, shell, kamada_kawai, random)
- `--color-scheme`: Node colors (default, degree, pagerank, depth)
- `--node-size`: Node sizing (constant, degree, pagerank)
- `--show-labels`/`--hide-labels`: Control label display
- `--verbose`: Enable verbose logging

### Examples

#### Basic Website Scraping
```bash
python -m web_graph_generator \
  --base-url https://example.com \
  --scrape \
  --max-depth 3 \
  --output-data website_links.json \
  --data-format json
```

#### Advanced Visualization
```bash
python -m web_graph_generator \
  --base-url https://example.com \
  --data-file website_links.json \
  --layout kamada_kawai \
  --color-scheme pagerank \
  --node-size degree \
  --output-image advanced_graph.svg
```

#### With URL Filtering
```bash
python -m web_graph_generator \
  --base-url https://example.com \
  --scrape \
  --skip-patterns skip_urls.txt \
  --no-cycles \
  --verbose
```

### Skip Patterns File Format

Create a text file with regex patterns (one per line):

```
# Skip admin pages
/admin/.*
/wp-admin/.*

# Skip file downloads
.*\.pdf$
.*\.zip$
.*\.exe$

# Skip specific query parameters
.*\?print=1
.*\?download=.*

# Skip social media links
.*facebook\.com.*
.*twitter\.com.*
```

## Package Structure

```
web_graph_generator/
├── __init__.py           # Package initialization
├── main.py              # Main entry point and CLI
├── url_handler.py       # URL normalization and filtering
├── scraper.py           # Web scraping functionality
├── graph_generator.py   # Graph creation and analysis
├── data_serializer.py   # Data serialization utilities
├── visualizer.py        # Graph visualization
└── requirements.txt     # Package dependencies
```

## API Usage

### Python API

```python
from web_graph_generator import WebScraper, GraphGenerator, GraphVisualizer
from web_graph_generator.url_handler import URLFilter

# Create URL filter
url_filter = URLFilter('skip_patterns.txt')

# Scrape website
scraper = WebScraper(
    base_url='https://example.com',
    max_depth=2,
    url_filter=url_filter
)
graph_data = scraper.scrape()

# Generate graph
generator = GraphGenerator(allow_cycles=True)
graph = generator.create_graph(graph_data)

# Create visualization
visualizer = GraphVisualizer(graph)
visualizer.create_visualization(
    'output.svg',
    layout='spring',
    color_scheme='pagerank'
)
```

### Data Serialization

```python
from web_graph_generator import DataSerializer

# Save data
DataSerializer.save_graph_data(graph_data, 'data.json', 'json')

# Load data
graph_data = DataSerializer.load_graph_data('data.json')

# Get file info
info = DataSerializer.get_file_info('data.json')
print(f"File contains {info['num_pages']} pages")
```

## Graph Analysis

The package provides comprehensive graph analysis capabilities:

```python
from web_graph_generator import GraphGenerator

generator = GraphGenerator()
graph = generator.create_graph(graph_data)

# Analyze graph properties
analysis = generator.analyze_graph(graph)
print(f"Graph density: {analysis['density']}")
print(f"Most linked pages: {analysis['most_linked_pages']}")

# Calculate PageRank
pagerank = generator.get_page_rank(graph)
top_pages = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]

# Find shortest paths
path = generator.find_shortest_path(graph, 'https://example.com', 'https://example.com/about')
```

## Visualization Options

### Layout Algorithms
- **spring**: Force-directed layout (default)
- **circular**: Nodes arranged in a circle
- **shell**: Nodes arranged in concentric circles
- **kamada_kawai**: Stress-minimization layout
- **random**: Random positioning

### Color Schemes
- **default**: Light blue nodes
- **degree**: Colors based on node degree
- **pagerank**: Colors based on PageRank values
- **depth**: Colors based on distance from central node

### Node Sizing
- **constant**: All nodes same size
- **degree**: Size based on node degree
- **pagerank**: Size based on PageRank values

## Use Cases

- **SEO Analysis**: Analyze internal link structures
- **Content Audit**: Identify orphaned pages and link gaps
- **Site Architecture**: Visualize information architecture
- **User Journey Mapping**: Understand navigation patterns
- **Competitor Analysis**: Study competitor site structures
- **Academic Research**: Web structure analysis

## Drupal Integration

This tool works well with Drupal 10.5.x multisites:

```bash
# Analyze Drupal site structure
python -m web_graph_generator \
  --base-url https://your-drupal-site.com \
  --scrape \
  --skip-patterns drupal_skip_patterns.txt \
  --max-depth 4 \
  --output-data drupal_links.json
```

Example `drupal_skip_patterns.txt`:
```
# Skip admin and system pages
/admin/.*
/user/.*
/system/.*
/core/.*

# Skip specific content types
/taxonomy/.*
/aggregator/.*

# Skip query parameters
.*\?destination=.*
.*\?page=.*
```

## Changelog

### Version 1.0.0
- Initial release
- Web scraping functionality
- Graph generation and analysis
- SVG visualization
- Multiple data formats
- Comprehensive CLI interface