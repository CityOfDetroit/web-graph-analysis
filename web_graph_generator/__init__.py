"""
Web Graph Generator Package

A Python package for generating directed graphs of web pages and their links.
Supports web scraping and visualization of link structures.
"""

from .graph_generator import GraphGenerator
from .scraper import WebScraper
from .url_handler import URLNormalizer, URLFilter
from .data_serializer import DataSerializer
from .visualizer import GraphVisualizer

__version__ = "1.0.0"
__author__ = "Maxwell Morgan <maxwell.morgan@detroitmi.gov>"
__description__ = "Generate directed graphs of web pages and links"

__all__ = [
    'GraphGenerator',
    'WebScraper', 
    'URLNormalizer',
    'URLFilter',
    'DataSerializer',
    'GraphVisualizer'
]