"""
Web Graph Generator Package

A Python package for generating interactive HTML visualizations of web pages and their links,
or Drupal taxonomy hierarchies. Supports web scraping and database-driven analysis.
"""

from .graph_generator import GraphGenerator
from .scraper import WebScraper, CSSFilter
from .drupal_taxonomy import DrupalTaxonomyExtractor, test_connection
from .url_handler import URLNormalizer, URLFilter
from .data_serializer import DataSerializer
from .visualizer import GraphVisualizer

__version__ = "1.0.0-alpha.1"
__author__ = "Maxwell Morgan <maxwell.morgan@detroitmi.gov>"
__description__ = "Generate interactive HTML graphs of web pages, links, and Drupal taxonomy hierarchies"

__all__ = [
    'GraphGenerator',
    'WebScraper', 
    'CSSFilter',
    'URLNormalizer',
    'URLFilter',
    'DataSerializer',
    'GraphVisualizer',
    'DrupalTaxonomyExtractor',
    'test_connection'
]