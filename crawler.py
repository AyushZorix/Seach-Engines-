import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import random

def fetch_page(url):
    """
    Fetch a page using a human-like delay and a polite User-Agent.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/118.0.5993.117 Safari/537.36 MyHumanLikeCrawler/1.0 (+https://example.com/contact)"
    }
    try:
        # Simulate human reading delay
        time.sleep(random.uniform(1, 3))
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"[ERROR] Failed to fetch {url}: {e}")
        return None

def extract_links(page_url, html):
    """
    Extract internal and external links from a page
    """
    soup = BeautifulSoup(html, "html.parser")
    base_domain = urlparse(page_url).netloc

    internal_links = set()
    external_links = set()

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()
        full_url = urljoin(page_url, href)
        parsed = urlparse(full_url)

        # Only http/https
        if parsed.scheme not in ("http", "https"):
            continue

        # Internal = same domain, External = different domain
        if parsed.netloc == base_domain:
            internal_links.add(full_url)
        else:
            external_links.add(full_url)

    return sorted(internal_links), sorted(external_links)

if __name__ == "__main__":
    url = "https://en.wikipedia.org/wiki/Web_crawler"
    print("=== BOT DECLARATION ===")
    print("This crawler acts politely, mimics human browsing delays,")
    print("and uses a clear User-Agent. For educational/research purposes.\n")

    html_content = fetch_page(url)
    if html_content:
        internal, external = extract_links(url, html_content)

        # Write output to a file
        with open("output.txt", "w", encoding="utf-8") as f:
            f.write(f"Internal Links ({len(internal)}):\n")
            for link in internal:
                f.write(link + "\n")

            f.write("\nExternal Links ({len(external)}):\n")
            for link in external:
                f.write(link + "\n")

        print(f"[INFO] Successfully wrote {len(internal)} internal and {len(external)} external links to output.txt")
