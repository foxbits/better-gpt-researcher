import os
import requests
from typing import Tuple, List


class Crawl4AI:
    """
    Scraper that uses local Crawl4AI service /crawl endpoint.
    Returns markdown content and images in GPT Researcher format.
    """

    def __init__(self, link, session=None):
        self.link = link
        self.session = session or requests.Session()
        self.base_url = self._get_crawl4ai_url()

    def _get_crawl4ai_url(self) -> str:
        """Get Crawl4AI server URL from environment or default to local container."""
        return os.environ.get("CRAWL4AI_API_URL", "http://crawl4ai:11235")

    def scrape(self) -> Tuple[str, List[dict], str]:
        """
        Call Crawl4AI /crawl endpoint and return content in GPT Researcher format.

        Returns:
            tuple: (content: str, image_urls: list, title: str)
        """
        try:
            response = self.session.post(
                f"{self.base_url}/crawl",
                json={"urls": [self.link]},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            if not data.get("success") or not data.get("results"):
                print("Crawl4AI crawl failed - no results")
                return "", [], ""

            result = data["results"][0]
            if not result.get("success"):
                print(f"Crawl4AI crawl failed: {result.get('error_message', 'unknown error')}")
                return "", [], ""

            markdown_data = result.get("markdown", {})
            content = markdown_data.get("raw_markdown", "")

            metadata = result.get("metadata", {})
            title = metadata.get("title", "") or ""

            media = result.get("media", {})
            images = media.get("images", [])

            image_urls = [
                {
                    "url": img.get("src", ""),
                    "score": img.get("score", 0),
                    "alt": img.get("alt", ""),
                    "description": img.get("description", "")
                }
                for img in images
                if img.get("src")
            ]

            return content, image_urls, title

        except requests.exceptions.Timeout:
            print("Crawl4AI crawl timeout")
            return "", [], ""
        except requests.exceptions.ConnectionError:
            print(f"Crawl4AI connection failed - is crawl4ai service available at {self.base_url}?")
            return "", [], ""
        except Exception as e:
            print(f"Crawl4AI error: {str(e)}")
            return "", [], ""
