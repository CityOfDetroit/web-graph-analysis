"""
Drupal taxonomy hierarchy extraction for web graph generation.

This module provides functionality to extract taxonomy hierarchies from a Drupal MySQL database
and convert them to the same format used by the web scraper.
"""

import logging
import sys
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse

try:
    import mysql.connector
    from mysql.connector import Error
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False
    mysql = None
    Error = Exception


class DrupalTaxonomyExtractor:
    """Extracts taxonomy hierarchies from Drupal MySQL database."""
    
    def __init__(self, db_config: Dict[str, str], base_term_id: int, 
                 max_depth: int = 5, base_url: str = ''):
        """
        Initialize Drupal taxonomy extractor.
        
        Args:
            db_config: Database connection configuration
            base_term_id: Starting taxonomy term ID
            max_depth: Maximum depth to traverse
            base_url: Base URL for the Drupal site (for URL construction)
        """
        if not MYSQL_AVAILABLE:
            raise ImportError("mysql-connector-python is required. Install with: pip install mysql-connector-python")
        
        self.db_config = db_config
        self.base_term_id = base_term_id
        self.max_depth = max_depth
        # Convert base URL to just domain and protocol, ensure it's set
        if base_url:
            parsed = urlparse(base_url)
            self.base_url = f"{parsed.scheme}://{parsed.netloc}"
        else:
            # Default to empty string - will be handled in _get_term_url
            self.base_url = 'https://detroitmi.gov'
        self.connection = None
        
        # Statistics
        self.terms_processed = 0
        self.aliases_found = 0
        self.default_urls_used = 0
        self.failed_queries = 0
    
    def connect(self) -> None:
        """
        Establish database connection.
        
        Raises:
            mysql.connector.Error: If connection fails
        """
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            if self.connection.is_connected():
                logging.info(f"Connected to Drupal database: {self.db_config.get('database', 'unknown')}")
            else:
                raise Error("Failed to establish connection")
        except Error as e:
            logging.error(f"Error connecting to MySQL database: {e}")
            raise
    
    def close(self) -> None:
        """Close database connection."""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logging.info("Database connection closed")
    
    def get_taxonomy_hierarchy(self) -> Dict[str, List[str]]:
        """
        Extract taxonomy hierarchy starting from base term.
        
        Returns:
            Dictionary mapping parent URLs to lists of child URLs
            
        Raises:
            Error: If database operations fail
        """
        if not self.connection:
            self.connect()
        
        try:
            hierarchy_data = {}
            visited = set()
            
            # Start with base term
            self._extract_hierarchy_recursive(
                self.base_term_id, 
                0, 
                hierarchy_data, 
                visited
            )
            
            self._log_statistics(hierarchy_data)
            return hierarchy_data
            
        except Error as e:
            logging.error(f"Error extracting taxonomy hierarchy: {e}")
            raise
    
    def _extract_hierarchy_recursive(self, term_id: int, depth: int, 
                                   hierarchy_data: Dict[str, List[str]], 
                                   visited: Set[int]) -> None:
        """
        Recursively extract taxonomy hierarchy.
        
        Args:
            term_id: Current term ID to process
            depth: Current depth level
            hierarchy_data: Dictionary to store hierarchy data
            visited: Set of already visited term IDs (cycle prevention)
        """
        if depth > self.max_depth or term_id in visited:
            return
        
        visited.add(term_id)
        self.terms_processed += 1
        
        # Get current term URL
        parent_url = self._get_term_url(term_id)
        if not parent_url:
            logging.warning(f"Could not get URL for term ID {term_id}")
            return
        
        # Get child terms
        child_term_ids = self._get_child_terms(term_id)
        child_urls = []
        
        for child_id in child_term_ids:
            child_url = self._get_term_url(child_id)
            if child_url:
                child_urls.append(child_url)
                
                # Recursively process child terms
                self._extract_hierarchy_recursive(
                    child_id, 
                    depth + 1, 
                    hierarchy_data, 
                    visited
                )
        
        # Store in hierarchy data
        hierarchy_data[parent_url] = child_urls
        
        logging.debug(f"Processed term {term_id} (depth {depth}): {len(child_urls)} children")
    
    def _get_term_url(self, term_id: int) -> Optional[str]:
        """
        Get URL for a taxonomy term, checking for alias first.
        
        Args:
            term_id: Taxonomy term ID
            
        Returns:
            Full URL for the term, or None if term not found
        """
        try:
            cursor = self.connection.cursor()
            
            # First, check if term exists, is published, and is English
            term_query = """
                SELECT name, vid 
                FROM taxonomy_term_field_data 
                WHERE tid = %s 
                AND default_langcode = 1 
                AND langcode = 'en'
                AND status = 1
                LIMIT 1
            """
            cursor.execute(term_query, (term_id,))
            term_result = cursor.fetchone()
            
            if not term_result:
                logging.warning(f"Term ID {term_id} not found, not published, or not English")
                return None
            
            term_name, vocabulary_id = term_result
            
            # Check for English URL alias specifically
            alias_query = """
                SELECT alias 
                FROM path_alias 
                WHERE path = %s 
                AND status = 1 
                AND langcode = 'en'
                ORDER BY id DESC 
                LIMIT 1
            """
            cursor.execute(alias_query, (f'/taxonomy/term/{term_id}',))
            alias_result = cursor.fetchone()
            
            if alias_result and alias_result[0]:
                # Found English alias
                alias = alias_result[0].lstrip('/')
                self.aliases_found += 1
                # Always include base_url
                url = f"{self.base_url}/{alias}"
                logging.debug(f"Found English alias for term {term_id} ({term_name}): {url}")
                return url
            else:
                # No English alias, use default taxonomy URL
                self.default_urls_used += 1
                # Always include base_url
                url = f"{self.base_url}/taxonomy/term/{term_id}"
                logging.debug(f"Using default URL for term {term_id} ({term_name}): {url}")
                return url
                
        except Error as e:
            logging.error(f"Error getting URL for term {term_id}: {e}")
            self.failed_queries += 1
            return None
        finally:
            if cursor:
                cursor.close()
    
    def _get_child_terms(self, parent_id: int) -> List[int]:
        """
        Get direct child terms for a given parent term.
        
        Args:
            parent_id: Parent term ID
            
        Returns:
            List of child term IDs
        """
        try:
            cursor = self.connection.cursor()
            
            # Use modern Drupal field-based parent relationship (English, published only)
            field_query = """
                SELECT DISTINCT t.entity_id 
                FROM taxonomy_term__parent t
                INNER JOIN taxonomy_term_field_data tf ON t.entity_id = tf.tid
                WHERE t.parent_target_id = %s 
                AND tf.default_langcode = 1 
                AND tf.langcode = 'en'
                AND tf.status = 1
                ORDER BY t.entity_id
            """
            
            cursor.execute(field_query, (parent_id,))
            field_results = cursor.fetchall()
            
            if field_results:
                child_ids = [row[0] for row in field_results]
                logging.debug(f"Found {len(child_ids)} published English children for term {parent_id}")
                return child_ids
            else:
                logging.debug(f"No published English children found for term {parent_id}")
                return []
            
        except Error as e:
            logging.error(f"Error getting child terms for {parent_id}: {e}")
            self.failed_queries += 1
            return []
        finally:
            if cursor:
                cursor.close()
    
    def get_term_info(self, term_id: int) -> Optional[Dict[str, str]]:
        """
        Get detailed information about a taxonomy term.
        
        Args:
            term_id: Taxonomy term ID
            
        Returns:
            Dictionary with term information or None if not found
        """
        try:
            cursor = self.connection.cursor()
            
            # Get term info (English, published only)
            query = """
                SELECT t.tid, t.name, t.description__value, t.vid
                FROM taxonomy_term_field_data t
                WHERE t.tid = %s 
                AND t.default_langcode = 1 
                AND t.langcode = 'en'
                AND t.status = 1
                LIMIT 1
            """
            cursor.execute(query, (term_id,))
            result = cursor.fetchone()
            
            if result:
                return {
                    'tid': result[0],
                    'name': result[1],
                    'description': result[2] or '',
                    'vocabulary_id': result[3],
                    'vocabulary_name': result[3]  # Use vid as name since vocab table may not exist
                }
            
            return None
            
        except Error as e:
            logging.error(f"Error getting info for term {term_id}: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
    
    def validate_term_exists(self, term_id: int) -> bool:
        """
        Validate that a term ID exists in the database.
        
        Args:
            term_id: Term ID to validate
            
        Returns:
            True if term exists
        """
        try:
            cursor = self.connection.cursor()
            
            query = """
                SELECT COUNT(*) 
                FROM taxonomy_term_field_data 
                WHERE tid = %s 
                AND default_langcode = 1 
                AND langcode = 'en'
                AND status = 1
            """
            cursor.execute(query, (term_id,))
            result = cursor.fetchone()
            
            exists = result and result[0] > 0
            if not exists:
                logging.error(f"Base term ID {term_id} does not exist, is not published, or is not English")
            
            return exists
            
        except Error as e:
            logging.error(f"Error validating term {term_id}: {e}")
            return False
        finally:
            if cursor:
                cursor.close()
    
    def _log_statistics(self, hierarchy_data: Dict[str, List[str]]) -> None:
        """Log extraction statistics."""
        total_terms = len(hierarchy_data)
        total_relationships = sum(len(children) for children in hierarchy_data.values())
        
        logging.info(f"Taxonomy extraction complete!")
        logging.info(f"  Terms processed: {self.terms_processed}")
        logging.info(f"  Terms in graph: {total_terms}")
        logging.info(f"  Parent-child relationships: {total_relationships}")
        logging.info(f"  URL aliases found: {self.aliases_found}")
        logging.info(f"  Default URLs used: {self.default_urls_used}")
        logging.info(f"  Failed queries: {self.failed_queries}")
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get extraction statistics.
        
        Returns:
            Dictionary with extraction statistics
        """
        return {
            'terms_processed': self.terms_processed,
            'aliases_found': self.aliases_found,
            'default_urls_used': self.default_urls_used,
            'failed_queries': self.failed_queries
        }


def test_connection(db_config: Dict[str, str]) -> bool:
    """
    Test database connection with given configuration.
    
    Args:
        db_config: Database connection configuration
        
    Returns:
        True if connection successful
    """
    if not MYSQL_AVAILABLE:
        logging.error("mysql-connector-python is required for database connections")
        return False
    
    try:
        connection = mysql.connector.connect(**db_config)
        if connection.is_connected():
            db_info = connection.get_server_info()
            logging.info(f"Successfully connected to MySQL server version {db_info}")
            
            cursor = connection.cursor()
            
            # Test basic Drupal table access
            cursor.execute("SHOW TABLES LIKE 'taxonomy_term_field_data'")
            if not cursor.fetchone():
                logging.error("taxonomy_term_field_data table not found - is this a Drupal database?")
                return False
            
            logging.info("Drupal taxonomy_term_field_data table found")
            
            # Test for parent relationship table (required)
            cursor.execute("SHOW TABLES LIKE 'taxonomy_term__parent'")
            if not cursor.fetchone():
                logging.error("taxonomy_term__parent table not found - this is required for hierarchy extraction")
                return False
            
            logging.info("taxonomy_term__parent table found")
            
            # Test for path alias table
            cursor.execute("SHOW TABLES LIKE 'path_alias'")
            if cursor.fetchone():
                logging.info("path_alias table found - URL aliases will be resolved")
            else:
                logging.warning("path_alias table not found - will use default taxonomy URLs")
            
            # Test sample data
            cursor.execute("""
                SELECT COUNT(*) FROM taxonomy_term_field_data 
                WHERE default_langcode = 1 AND langcode = 'en' AND status = 1
            """)
            count_result = cursor.fetchone()
            english_terms = count_result[0] if count_result else 0
            logging.info(f"Found {english_terms} published English taxonomy terms")
            
            if english_terms == 0:
                logging.warning("No published English taxonomy terms found")
                return False
            
            return True
                
    except Error as e:
        logging.error(f"Error testing database connection: {e}")
        return False
    finally:
        if 'connection' in locals() and connection.is_connected():
            connection.close()