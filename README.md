# Web Graph Generator

A Python package for generating directed graphs of web pages and their links. This tool can scrape websites or load existing data to create visualizations and analyze link structures.

## Features

- **Web Scraping**: Breadth-first traversal of websites with configurable depth
- **Graph Generation**: Creates weighted directed graphs using NetworkX
- **URL Filtering**: Regex-based URL filtering with customizable patterns
- **CSS Element Filtering**: Skip links within specific HTML elements using CSS selectors
- **Multiple Output Formats**: Static SVG and interactive HTML visualizations
- **Rich Visualizations**: Multiple layout algorithms, color schemes, and interactive features
- **Self-Link Filtering**: Automatically filters out same-page links (skip links, anchors, etc.)
- **Cycle Handling**: Optional cycle detection and prevention
- **Modular Design**: Clean, extensible architecture with separate modules

## Installation

### From Source

```bash
git clone https://github.com/CityofDetroit/web-graph-analysis.git
cd web-graph-analysis
pip install -e .
```

## Quick Start

### Scrape a Website (Interactive)

```bash
python -m web_graph_generator --base-url https://example.com --scrape --max-depth 2 --output-data graph_data.json --data-format json --output-image graph.html
```

### Load Existing Data (Static)

```bash
python -m web_graph_generator --base-url https://example.com --data-file graph_data.json --output-image graph.svg
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
- `--skip-selectors`: File with CSS selectors for HTML elements to skip when extracting links

#### Output Options
- `--output-data`: Path for serialized data output
- `--data-format`: Format for data (pickle or json)
- `--output-image`: Path for graph image (.svg for static, .html for interactive)

#### Visualization Options
- `--layout`: Layout algorithm (spring, circular, shell, kamada_kawai, random)
- `--color-scheme`: Node colors (default, degree, pagerank, depth)
- `--node-size`: Node sizing (constant, degree, pagerank)
- `--show-labels`/`--hide-labels`: Control label display
- `--verbose`: Enable verbose logging

### Examples

#### Interactive Website Analysis
```bash
python -m web_graph_generator \
  --base-url https://example.com \
  --scrape \
  --max-depth 3 \
  --output-data website_links.json \
  --data-format json \
  --output-image interactive_graph.html
```

#### Static Visualization with Advanced Options
```bash
python -m web_graph_generator \
  --base-url https://example.com \
  --data-file website_links.json \
  --layout kamada_kawai \
  --color-scheme pagerank \
  --node-size degree \
  --output-image advanced_graph.svg
```

#### With URL and CSS Filtering
```bash
python -m web_graph_generator \
  --base-url https://example.com \
  --scrape \
  --skip-patterns skip_urls.txt \
  --skip-selectors skip_elements.txt \
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

### CSS Selectors File Format

Create a text file with CSS selectors (one per line) to skip HTML elements when extracting links:

```
nav
.navigation
#main-menu

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
from web_graph_generator.scraper import CSSFilter

# Create URL filter
url_filter = URLFilter('skip_patterns.txt')

# Create CSS filter
css_filter = CSSFilter('skip_selectors.txt')

# Scrape website
scraper = WebScraper(
    base_url='https://example.com',
    max_depth=2,
    url_filter=url_filter,
    css_filter=css_filter
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

## Visualization Types

### Interactive HTML Graphs (.html)
- **Click to visit**: Nodes are clickable and open the actual web page
- **Hover details**: Full URL, in-degree, out-degree, and other metrics
- **Zoom and pan**: Navigate large site structures easily
- **Abbreviated URLs**: Smart URL abbreviation for better readability (e.g., `/about/.../team`)
- **Professional styling**: Clean, modern appearance suitable for presentations

### Static SVG Graphs (.svg)
- **Publication ready**: High-quality vector graphics for reports and documents
- **Lightweight**: Smaller file sizes for embedding in documents
- **Print friendly**: Scales perfectly for printed materials
- **Traditional labels**: Simple path-based node labels

## Output Format Selection

The output format is automatically detected from the file extension:

```bash
# Interactive HTML visualization
--output-image graph.html

# Static SVG visualization  
--output-image graph.svg

# Static PNG visualization
--output-image graph.png
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

- **SEO Analysis**: Analyze internal link structures with interactive exploration
- **Content Audit**: Identify orphaned pages and link gaps with clickable navigation
- **Site Architecture**: Visualize information architecture with abbreviated URL paths
- **User Journey Mapping**: Understand navigation patterns through interactive graphs
- **Competitor Analysis**: Study competitor site structures with exportable data
- **Academic Research**: Web structure analysis with both static and interactive outputs

## Drupal Integration

This tool works well with Drupal 10.5.x multisites:

```bash
# Analyze Drupal site structure (interactive)
python -m web_graph_generator \
  --base-url https://your-drupal-site.com \
  --scrape \
  --skip-patterns example_skip_url_regex.txt \
  --skip-selectors example_skip_css_selectors.txt \
  --max-depth 4 \
  --output-data drupal_links.json \
  --output-image drupal_graph.html
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
## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
- Open an issue on GitHub
- Check the documentation
- Review existing issues for solutions

## Changelog

### Version 1.0.0
- Initial release
- Web scraping functionality
- Graph generation and analysis
- SVG visualization
- Multiple data formats
- Comprehensive CLI interface