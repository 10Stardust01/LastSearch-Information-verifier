#!/usr/bin/env python3
"""
Bengaluru Civic Sources Crawler

Crawls official Bengaluru civic websites and indexes documents into Elasticsearch.
Uses the .crawler.yml configuration for sources and extraction rules.
"""

import asyncio
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests
import yaml
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CivicCrawler:
    def __init__(self, config_path: str = ".crawler.yml"):
        self.config = self.load_config(config_path)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Bengaluru-Misinfo-Layer-Crawler/1.0 (civic-fact-checking)'
        })

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load crawler configuration from YAML file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def should_crawl_url(self, url: str) -> bool:
        """Check if URL should be crawled based on crawl rules."""
        for rule in self.config['crawl_rules']:
            pattern = rule['pattern']
            if rule['type'] == 'begins':
                if url.startswith(pattern):
                    return rule['policy'] == 'allow'
            elif rule['type'] == 'regex':
                if re.match(pattern, url):
                    return rule['policy'] == 'allow'
        return False

    def get_extraction_rules(self, url: str) -> Optional[Dict[str, Any]]:
        """Get extraction rules for a URL."""
        for rule in self.config['extraction_rules']:
            for url_filter in rule['url_filters']:
                pattern = url_filter['pattern']
                if url_filter['type'] == 'begins' and url.startswith(pattern):
                    return rule
        return None

    def extract_field(self, soup: BeautifulSoup, field_config: Dict[str, Any]) -> str:
        """Extract a field from HTML using selector or fixed value."""
        if 'value' in field_config:
            return field_config['value']

        selector = field_config['selector']
        elements = soup.select(selector)

        if not elements:
            return ""

        element = elements[0]

        # Extract text content
        if field_config.get('source') == 'html':
            return element.get_text(strip=True)
        else:
            return element.get_text(strip=True)

    def crawl_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Crawl a single URL and extract document data."""
        try:
            logger.info(f"Crawling: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            rules = self.get_extraction_rules(url)

            if not rules:
                logger.warning(f"No extraction rules for {url}")
                return None

            document = {
                'source_url': url,
                'crawled_at': datetime.now(timezone.utc).isoformat(),
                'content_type': 'text/html'
            }

            # Extract fields according to rules
            for field in rules['fields']:
                field_name = field['field_name']
                document[field_name] = self.extract_field(soup, field)

            # Set source_tier based on source_name
            source_name = document.get('source_name', '')
            if source_name in ['BMRCL', 'BBMP', 'BWSSB', 'BESCOM', 'BDA', 'PIB Karnataka']:
                document['source_tier'] = 'official'
            else:
                document['source_tier'] = 'news'

            logger.info(f"Extracted document: {document.get('title', 'No title')}")
            return document

        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")
            return None

    def crawl_source_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Crawl multiple URLs and return extracted documents."""
        documents = []
        for url in urls:
            if self.should_crawl_url(url):
                doc = self.crawl_url(url)
                if doc:
                    documents.append(doc)
            else:
                logger.info(f"Skipping disallowed URL: {url}")
        return documents

    def index_to_elasticsearch(self, documents: List[Dict[str, Any]]) -> None:
        """Index documents into Elasticsearch."""
        try:
            from elasticsearch import Elasticsearch
            import os

            # Load environment variables
            es_url = os.getenv('ELASTICSEARCH_URL', 'http://localhost:9200')
            es_api_key = os.getenv('ELASTICSEARCH_API_KEY')
            index_name = os.getenv('ELASTICSEARCH_INDEX', 'bengaluru-sources')

            if not es_api_key:
                raise ValueError("ELASTICSEARCH_API_KEY environment variable required")

            es = Elasticsearch(es_url, api_key=es_api_key)

            for doc in documents:
                try:
                    es.index(index=index_name, document=doc)
                    logger.info(f"Indexed: {doc.get('title', 'No title')}")
                except Exception as e:
                    logger.error(f"Failed to index document: {e}")

        except ImportError:
            logger.error("Elasticsearch dependencies not installed. Run: pip install -e .[live]")
        except Exception as e:
            logger.error(f"Elasticsearch indexing error: {e}")

def main():
    """Main crawler entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Crawl Bengaluru civic sources")
    parser.add_argument('--urls', nargs='+', help='Specific URLs to crawl')
    parser.add_argument('--config', default='.crawler.yml', help='Crawler config file')
    parser.add_argument('--dry-run', action='store_true', help='Extract but do not index')

    args = parser.parse_args()

    crawler = CivicCrawler(args.config)

    if args.urls:
        urls = args.urls
    else:
        # Default URLs to crawl (recent press releases, notices)
        urls = [
            "https://www.bmrcl.co.in/press-releases/",
            "https://bbmp.gov.in/en/circulars",
            "https://bwssb.gov.in/notices",
            "https://bescom.karnataka.gov.in/news",
            "https://pib.gov.in/PressReleasePage.aspx?PRID=123456",  # Example
        ]

    logger.info(f"Starting crawl of {len(urls)} URLs")
    documents = crawler.crawl_source_urls(urls)

    if args.dry_run:
        print(json.dumps(documents, indent=2, ensure_ascii=False))
    else:
        crawler.index_to_elasticsearch(documents)

    logger.info(f"Crawled {len(documents)} documents successfully")

if __name__ == "__main__":
    main()