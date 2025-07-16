"""
URL handling utilities for web graph generation.

This module provides URL normalization and filtering functionality.
"""

import logging
import re
import sys
from typing import List, Optional
from urllib.parse import urlparse, urlunparse


class URLNormalizer:
    """Handles URL normalization and validation."""
    
    @staticmethod
    def normalize_url(url: str) -> str:
        """
        Normalize URL by removing query parameters, fragments, and trailing slashes.
        
        Args:
            url: Raw URL string
            
        Returns:
            Normalized URL string
        """
        parsed = urlparse(url)
        # Remove query parameters and fragments
        normalized = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path.rstrip('/') if parsed.path != '/' else '/',
            '',  # params
            '',  # query
            ''   # fragment
        ))
        return normalized
    
    @staticmethod
    def is_valid_url(url: str, base_domain: str = None) -> bool:
        """
        Check if URL is valid. Optionally restrict to base domain.
        
        Args:
            url: URL to validate
            base_domain: Base domain to restrict crawling to (optional)
            
        Returns:
            True if URL is valid (and within domain if base_domain specified)
        """
        try:
            parsed = urlparse(url)
            
            # Must have scheme and netloc
            if not parsed.scheme in ('http', 'https') or not parsed.netloc:
                return False
            
            # If base_domain is specified, restrict to that domain
            if base_domain is not None:
                return parsed.netloc == base_domain and parsed.path
            
            # Otherwise, any valid HTTP/HTTPS URL is acceptable
            return True
            
        except Exception:
            return False


class URLFilter:
    """Handles URL filtering based on regex patterns."""
    
    def __init__(self, skip_patterns_file: Optional[str] = None):
        """
        Initialize URL filter with optional skip patterns file.
        
        Args:
            skip_patterns_file: Path to file containing regex patterns to skip
        """
        self.skip_patterns: List[re.Pattern] = []
        if skip_patterns_file:
            self.load_skip_patterns(skip_patterns_file)
    
    def load_skip_patterns(self, patterns_file: str) -> None:
        """
        Load regex patterns from file.
        
        Args:
            patterns_file: Path to file containing regex patterns
            
        Raises:
            SystemExit: If file not found or contains invalid regex
        """
        try:
            with open(patterns_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        try:
                            self.skip_patterns.append(re.compile(line))
                        except re.error as e:
                            logging.error(f"Invalid regex pattern at line {line_num}: {line} - {e}")
                            sys.exit(1)
            logging.info(f"Loaded {len(self.skip_patterns)} skip patterns")
        except FileNotFoundError:
            logging.error(f"Skip patterns file not found: {patterns_file}")
            sys.exit(1)
    
    def should_skip(self, url: str) -> bool:
        """
        Check if URL should be skipped based on patterns.
        
        Args:
            url: URL to check against patterns
            
        Returns:
            True if URL matches any skip pattern
        """
        for pattern in self.skip_patterns:
            if pattern.search(url):
                return True
        return False
    
    def add_pattern(self, pattern: str) -> None:
        """
        Add a regex pattern to the skip list.
        
        Args:
            pattern: Regex pattern string
            
        Raises:
            re.error: If pattern is invalid
        """
        compiled_pattern = re.compile(pattern)
        self.skip_patterns.append(compiled_pattern)
    
    def get_patterns_count(self) -> int:
        """
        Get the number of loaded skip patterns.
        
        Returns:
            Number of skip patterns
        """
        return len(self.skip_patterns)
    
    def get_patterns_html(self) -> str:
        """
        Returns a string representation of all loaded skip patterns.
        """
        if not self.skip_patterns:
            return "None"

        return "<br>".join(pattern.pattern for pattern in self.skip_patterns)