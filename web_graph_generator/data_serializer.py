"""
Data serialization utilities for web graph generation.

This module provides functionality to save and load graph data in various formats.
"""

import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Union

import networkx as nx


class DataSerializer:
    """Handles serialization and deserialization of graph data."""
    
    SUPPORTED_FORMATS = ['pickle', 'json']
    PICKLE_EXTENSIONS = ['.pkl', '.pickle']
    JSON_EXTENSIONS = ['.json']
    
    @classmethod
    def save_graph_data(cls, data: Dict[str, List[str]], filepath: str, 
                       format_type: str) -> None:
        """
        Save graph data to file.
        
        Args:
            data: Graph data dictionary mapping URLs to lists of linked URLs
            filepath: Output file path
            format_type: 'pickle' or 'json'
            
        Raises:
            ValueError: If format_type is not supported
            IOError: If file cannot be written
        """
        if format_type not in cls.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {format_type}. "
                           f"Supported formats: {cls.SUPPORTED_FORMATS}")
        
        try:
            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            if format_type == 'pickle':
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            elif format_type == 'json':
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Saved graph data to {filepath} in {format_type} format")
            
        except Exception as e:
            logging.error(f"Failed to save graph data: {e}")
            raise IOError(f"Could not save data to {filepath}: {e}")
    
    @classmethod
    def load_graph_data(cls, filepath: str) -> Dict[str, List[str]]:
        """
        Load graph data from file, auto-detecting format from extension.
        
        Args:
            filepath: Input file path
            
        Returns:
            Graph data dictionary mapping URLs to lists of linked URLs
            
        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file format cannot be determined or data is invalid
            IOError: If file cannot be read
        """
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        if not path.is_file():
            raise ValueError(f"Path is not a file: {filepath}")
        
        try:
            # Try to determine format from extension
            format_type = cls._detect_format(filepath)
            
            if format_type == 'json':
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif format_type == 'pickle':
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
            else:
                # Auto-detect by trying formats
                data = cls._auto_detect_and_load(filepath)
            
            # Validate data structure
            cls._validate_graph_data(data)
            
            logging.info(f"Loaded graph data from {filepath} ({format_type} format)")
            return data
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {filepath}: {e}")
        except pickle.UnpicklingError as e:
            raise ValueError(f"Invalid pickle format in {filepath}: {e}")
        except Exception as e:
            logging.error(f"Failed to load graph data: {e}")
            raise IOError(f"Could not load data from {filepath}: {e}")
    
    @classmethod
    def _detect_format(cls, filepath: str) -> str:
        """
        Detect file format from extension.
        
        Args:
            filepath: File path
            
        Returns:
            Format type ('json', 'pickle', or 'unknown')
        """
        path = Path(filepath)
        suffix = path.suffix.lower()
        
        if suffix in cls.JSON_EXTENSIONS:
            return 'json'
        elif suffix in cls.PICKLE_EXTENSIONS:
            return 'pickle'
        else:
            return 'unknown'
    
    @classmethod
    def _auto_detect_and_load(cls, filepath: str) -> Dict[str, List[str]]:
        """
        Auto-detect format and load data.
        
        Args:
            filepath: File path
            
        Returns:
            Loaded data
            
        Raises:
            ValueError: If neither format works
        """
        # Try JSON first (human-readable)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logging.info(f"Auto-detected JSON format for {filepath}")
            return data
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
        
        # Try pickle
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            logging.info(f"Auto-detected pickle format for {filepath}")
            return data
        except (pickle.UnpicklingError, EOFError):
            pass
        
        raise ValueError(f"Could not determine format for {filepath}")
    
    @classmethod
    def _validate_graph_data(cls, data: any) -> None:
        """
        Validate that loaded data has the expected structure.
        
        Args:
            data: Loaded data to validate
            
        Raises:
            ValueError: If data structure is invalid
        """
        if not isinstance(data, dict):
            raise ValueError("Graph data must be a dictionary")
        
        for url, links in data.items():
            if not isinstance(url, str):
                raise ValueError(f"URL keys must be strings, got {type(url)}")
            
            if not isinstance(links, list):
                raise ValueError(f"Link values must be lists, got {type(links)} for {url}")
            
            for link in links:
                if not isinstance(link, str):
                    raise ValueError(f"Links must be strings, got {type(link)} in {url}")
    
    @classmethod
    def save_networkx_graph(cls, graph: nx.DiGraph, filepath: str, 
                           format_type: str = 'graphml') -> None:
        """
        Save NetworkX graph to file.
        
        Args:
            graph: NetworkX DiGraph to save
            filepath: Output file path
            format_type: Format type ('graphml', 'gexf', 'pickle')
            
        Raises:
            ValueError: If format_type is not supported
            IOError: If file cannot be written
        """
        supported_nx_formats = ['graphml', 'gexf', 'pickle']
        
        if format_type not in supported_nx_formats:
            raise ValueError(f"Unsupported NetworkX format: {format_type}. "
                           f"Supported formats: {supported_nx_formats}")
        
        try:
            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            if format_type == 'graphml':
                nx.write_graphml(graph, filepath)
            elif format_type == 'gexf':
                nx.write_gexf(graph, filepath)
            elif format_type == 'pickle':
                nx.write_gpickle(graph, filepath)
            
            logging.info(f"Saved NetworkX graph to {filepath} in {format_type} format")
            
        except Exception as e:
            logging.error(f"Failed to save NetworkX graph: {e}")
            raise IOError(f"Could not save graph to {filepath}: {e}")
    
    @classmethod
    def load_networkx_graph(cls, filepath: str) -> nx.DiGraph:
        """
        Load NetworkX graph from file, auto-detecting format.
        
        Args:
            filepath: Input file path
            
        Returns:
            NetworkX DiGraph
            
        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file format cannot be determined
            IOError: If file cannot be read
        """
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"Graph file not found: {filepath}")
        
        try:
            suffix = path.suffix.lower()
            
            if suffix == '.graphml':
                graph = nx.read_graphml(filepath)
            elif suffix == '.gexf':
                graph = nx.read_gexf(filepath)
            elif suffix in ['.pkl', '.pickle']:
                graph = nx.read_gpickle(filepath)
            else:
                # Try to auto-detect
                graph = cls._auto_detect_and_load_nx(filepath)
            
            # Ensure it's a directed graph
            if not isinstance(graph, nx.DiGraph):
                graph = graph.to_directed()
            
            logging.info(f"Loaded NetworkX graph from {filepath}")
            return graph
            
        except Exception as e:
            logging.error(f"Failed to load NetworkX graph: {e}")
            raise IOError(f"Could not load graph from {filepath}: {e}")
    
    @classmethod
    def _auto_detect_and_load_nx(cls, filepath: str) -> nx.DiGraph:
        """
        Auto-detect NetworkX format and load graph.
        
        Args:
            filepath: File path
            
        Returns:
            NetworkX DiGraph
            
        Raises:
            ValueError: If no format works
        """
        # Try formats in order of preference
        formats = [
            ('graphml', nx.read_graphml),
            ('gexf', nx.read_gexf),
            ('pickle', nx.read_gpickle)
        ]
        
        for format_name, loader in formats:
            try:
                graph = loader(filepath)
                logging.info(f"Auto-detected {format_name} format for {filepath}")
                return graph
            except Exception:
                continue
        
        raise ValueError(f"Could not determine NetworkX format for {filepath}")
    
    @classmethod
    def convert_format(cls, input_filepath: str, output_filepath: str, 
                      output_format: str) -> None:
        """
        Convert graph data from one format to another.
        
        Args:
            input_filepath: Input file path
            output_filepath: Output file path
            output_format: Target format ('pickle', 'json')
            
        Raises:
            ValueError: If output format is not supported
            IOError: If conversion fails
        """
        if output_format not in cls.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        try:
            # Load data
            data = cls.load_graph_data(input_filepath)
            
            # Save in new format
            cls.save_graph_data(data, output_filepath, output_format)
            
            logging.info(f"Converted {input_filepath} to {output_filepath} "
                        f"in {output_format} format")
            
        except Exception as e:
            logging.error(f"Failed to convert format: {e}")
            raise IOError(f"Could not convert {input_filepath} to {output_format}: {e}")
    
    @classmethod
    def get_file_info(cls, filepath: str) -> Dict[str, any]:
        """
        Get information about a graph data file.
        
        Args:
            filepath: File path to analyze
            
        Returns:
            Dictionary with file information
        """
        path = Path(filepath)
        
        if not path.exists():
            return {'exists': False, 'error': 'File not found'}
        
        try:
            info = {
                'exists': True,
                'size_bytes': path.stat().st_size,
                'detected_format': cls._detect_format(filepath),
                'extension': path.suffix.lower()
            }
            
            # Try to load and get basic stats
            try:
                data = cls.load_graph_data(filepath)
                info['num_pages'] = len(data)
                info['total_links'] = sum(len(links) for links in data.values())
                info['avg_links_per_page'] = info['total_links'] / info['num_pages'] if info['num_pages'] > 0 else 0
                info['loadable'] = True
            except Exception as e:
                info['loadable'] = False
                info['load_error'] = str(e)
            
            return info
            
        except Exception as e:
            return {'exists': True, 'error': str(e)}
    
    @classmethod
    def create_backup(cls, filepath: str, backup_suffix: str = '.bak') -> str:
        """
        Create a backup of a data file.
        
        Args:
            filepath: Original file path
            backup_suffix: Suffix to add to backup file
            
        Returns:
            Path to backup file
            
        Raises:
            IOError: If backup creation fails
        """
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"Cannot backup non-existent file: {filepath}")
        
        backup_path = path.with_suffix(path.suffix + backup_suffix)
        
        try:
            backup_path.write_bytes(path.read_bytes())
            logging.info(f"Created backup: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logging.error(f"Failed to create backup: {e}")
            raise IOError(f"Could not create backup of {filepath}: {e}")