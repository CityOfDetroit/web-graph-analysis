"""
Web scraping functionality for link graph generation.

This module provides web scraping capabilities with breadth-first traversal
and link extraction from HTML pages.
"""

import logging
import sys
import time
from collections import deque
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .url_handler import URLNormalizer, URLFilter


class CSSFilter:
    """Handles CSS selector-based filtering for HTML elements."""
    
    def __init__(self, skip_selectors_file: Optional[str] = None):
        """
        Initialize CSS filter with optional skip selectors file.
        
        Args:
            skip_selectors_file: Path to file containing CSS selectors to skip
        """
        self.skip_selectors: List[str] = []
        if skip_selectors_file:
            self.load_skip_selectors(skip_selectors_file)
    
    def load_skip_selectors(self, selectors_file: str) -> None:
        """
        Load CSS selectors from file.
        
        Args:
            selectors_file: Path to file containing CSS selectors
            
        Raises:
            SystemExit: If file not found
        """
        try:
            with open(selectors_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Basic validation of CSS selector
                        if self._is_valid_selector(line):
                            self.skip_selectors.append(line)
                        else:
                            logging.warning(f"Invalid CSS selector at line {line_num}: {line}")
            logging.info(f"Loaded {len(self.skip_selectors)} CSS skip selectors")
        except FileNotFoundError:
            logging.error(f"CSS selectors file not found: {selectors_file}")
            sys.exit(1)
    
    def _is_valid_selector(self, selector: str) -> bool:
        """
        Basic validation of CSS selector syntax.
        
        Args:
            selector: CSS selector string
            
        Returns:
            True if selector appears valid
        """
        # Basic checks - not exhaustive but catches common errors
        if not selector:
            return False
        
        # Check for balanced brackets
        if selector.count('[') != selector.count(']'):
            return False
        
        if selector.count('(') != selector.count(')'):
            return False
        
        # Check for invalid characters at start
        if selector.startswith((',', '>', '+', '~')):
            return False
            
        return True
    
    def should_skip_element(self, element, soup) -> bool:
        """
        Check if element should be skipped based on CSS selectors.
        
        Args:
            element: BeautifulSoup element to check
            soup: BeautifulSoup object for CSS selection
            
        Returns:
            True if element should be skipped
        """
        for selector in self.skip_selectors:
            try:
                # Check if the element is within any element matching the selector
                matching_elements = soup.select(selector)
                for matching_element in matching_elements:
                    if self._is_element_within(element, matching_element):
                        return True
            except Exception as e:
                logging.warning(f"Error applying CSS selector '{selector}': {e}")
                continue
        
        return False
    
    def _is_element_within(self, element, parent) -> bool:
        """
        Check if element is within parent element.
        
        Args:
            element: Element to check
            parent: Potential parent element
            
        Returns:
            True if element is within parent
        """
        current = element
        while current:
            if current == parent:
                return True
            current = current.parent
        return False
    
    def get_selectors_count(self) -> int:
        """
        Get the number of loaded CSS selectors.
        
        Returns:
            Number of CSS selectors
        """
        return len(self.skip_selectors)


class WebScraper:
    """Handles web scraping and link extraction."""
    
    def __init__(self, base_url: str, max_depth: int, url_filter: URLFilter, 
                 css_filter: Optional['CSSFilter'] = None,
                 allow_cycles: bool = True, delay: float = 1.0, timeout: int = 10):
        """
        Initialize web scraper.
        
        Args:
            base_url: Starting URL for crawling
            max_depth: Maximum depth to crawl
            url_filter: URL filter instance
            css_filter: CSS filter instance for skipping elements
            allow_cycles: Whether to allow cycles in the graph
            delay: Delay between requests in seconds
            timeout: Request timeout in seconds
        """
        self.base_url = URLNormalizer.normalize_url(base_url)
        self.max_depth = max_depth
        self.url_filter = url_filter
        self.css_filter = css_filter
        self.allow_cycles = allow_cycles
        self.delay = delay
        self.timeout = timeout
        self.base_domain = urlparse(self.base_url).netloc
        
        # Initialize session with headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; WebGraphGenerator/1.0)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        })
        
        # Statistics
        self.pages_scraped = 0
        self.failed_requests = 0
        self.total_links_found = 0
        self.links_skipped_by_css = 0
        self.self_links_filtered = 0
    
    def extract_links(self, html_content: str, current_url: str) -> List[str]:
        """
        Extract all links from HTML content.
        
        Args:
            html_content: HTML content to parse
            current_url: Current page URL for resolving relative links
            
        Returns:
            List of normalized URLs
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            links = []
            
            # Extract links from <a> tags
            for link in soup.find_all('a', href=True):
                href = link['href']
                # Skip empty hrefs and javascript/mailto links
                if not href or href.startswith(('javascript:', 'mailto:', 'tel:')):
                    continue
                
                # Skip links based on CSS selectors
                if self.css_filter and self.css_filter.should_skip_element(link, soup):
                    self.links_skipped_by_css += 1
                    logging.debug(f"Skipping link due to CSS selector: {href}")
                    continue
                
                # Resolve relative URLs
                absolute_url = urljoin(current_url, href)
                normalized_url = URLNormalizer.normalize_url(absolute_url)
                
                # Skip self-links (links to the same page after normalization)
                normalized_current_url = URLNormalizer.normalize_url(current_url)
                if normalized_url == normalized_current_url:
                    self.self_links_filtered += 1
                    logging.debug(f"Filtering self-link: {href} -> {normalized_url}")
                    continue
                
                # Filter URLs
                if (URLNormalizer.is_valid_url(normalized_url, self.base_domain) and
                    not self.url_filter.should_skip(normalized_url)):
                    links.append(normalized_url)
            
            # Remove duplicates while preserving order
            unique_links = []
            seen = set()
            for link in links:
                if link not in seen:
                    unique_links.append(link)
                    seen.add(link)
            
            self.total_links_found += len(unique_links)
            return unique_links
            
        except Exception as e:
            logging.error(f"Error extracting links from {current_url}: {e}")
            return []
    
    def fetch_page(self, url: str) -> Optional[str]:
        """
        Fetch a single web page.
        
        Args:
            url: URL to fetch
            
        Returns:
            HTML content or None if failed
        """
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if not content_type.startswith('text/html'):
                logging.warning(f"Skipping non-HTML content at {url}: {content_type}")
                return None
            
            return response.text
            
        except requests.exceptions.Timeout:
            logging.warning(f"Timeout fetching {url}")
            self.failed_requests += 1
            return None
        except requests.exceptions.ConnectionError:
            logging.warning(f"Connection error fetching {url}")
            self.failed_requests += 1
            return None
        except requests.exceptions.HTTPError as e:
            logging.warning(f"HTTP error fetching {url}: {e}")
            self.failed_requests += 1
            return None
        except requests.RequestException as e:
            logging.warning(f"Request error fetching {url}: {e}")
            self.failed_requests += 1
            return None
    
    def scrape(self) -> Dict[str, List[str]]:
        """
        Scrape web pages and extract links using BFS.
        
        Returns:
            Dictionary mapping URLs to lists of linked URLs
        """
        graph_data = {}
        visited = set()
        queue = deque([(self.base_url, 0)])  # (url, depth)
        
        logging.info(f"Starting scrape from {self.base_url} (max depth: {self.max_depth})")
        
        while queue:
            current_url, depth = queue.popleft()
            
            if depth > self.max_depth:
                continue
                
            if not self.allow_cycles and current_url in visited:
                continue
                
            if current_url in visited:
                # For cycles allowed, still track the link but don't re-scrape
                if current_url not in graph_data:
                    graph_data[current_url] = []
                continue
            
            visited.add(current_url)
            self.pages_scraped += 1
            
            logging.info(f"Scraping (depth {depth}): {current_url}")
            
            # Fetch page content
            html_content = self.fetch_page(current_url)
            if html_content is None:
                graph_data[current_url] = []
                continue
            
            # Extract links
            links = self.extract_links(html_content, current_url)
            graph_data[current_url] = links
            
            logging.debug(f"Found {len(links)} links on {current_url}")
            
            # Add new links to queue for next depth level
            if depth < self.max_depth:
                for link in links:
                    if self.allow_cycles or link not in visited:
                        queue.append((link, depth + 1))
            
            # Be respectful with delays
            if self.delay > 0:
                time.sleep(self.delay)
        
        self._log_statistics(graph_data)
        return graph_data
    
    def _log_statistics(self, graph_data: Dict[str, List[str]]) -> None:
        """Log scraping statistics."""
        logging.info(f"Scraping complete!")
        logging.info(f"  Pages scraped: {self.pages_scraped}")
        logging.info(f"  Failed requests: {self.failed_requests}")
        logging.info(f"  Total links found: {self.total_links_found}")
        logging.info(f"  Links skipped by CSS selectors: {self.links_skipped_by_css}")
        logging.info(f"  Self-links filtered: {self.self_links_filtered}")
        logging.info(f"  Unique pages in graph: {len(graph_data)}")
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get scraping statistics.
        
        Returns:
            Dictionary with scraping statistics
        """
        return {
            'pages_scraped': self.pages_scraped,
            'failed_requests': self.failed_requests,
            'total_links_found': self.total_links_found,
            'links_skipped_by_css': self.links_skipped_by_css,
            'self_links_filtered': self.self_links_filtered
        }
    
    def close(self) -> None:
        """Close the session."""
        self.session.close()