"""
Fill empty url2text fields in knowledge store JSON files.

Reads existing knowledge store files and fetches text content from URLs
where url2text is empty. Uses Playwright for JavaScript-heavy sites like
Facebook, Instagram, Twitter. Saves filled results to a new directory.

Usage:
    python src/retrieval/fill_url2text.py \
        --input_dir dataset/AVerImaTeC_Shared_Task/Knowledge_Store/val/text_related/image_related_store_text_val \
        --output_dir dataset/AVerImaTeC_Shared_Task/Knowledge_Store/val/text_related/image_related_store_text_val_filled
"""

import os
import json
import argparse
import time
from tqdm import tqdm
from typing import List, Any
import requests

# Domains that require Playwright (JavaScript rendering)
PLAYWRIGHT_DOMAINS = {
    'facebook.com', 'www.facebook.com', 'm.facebook.com',
    'instagram.com', 'www.instagram.com',
    'twitter.com', 'www.twitter.com', 'x.com', 'www.x.com',
    'tiktok.com', 'www.tiktok.com',
    'linkedin.com', 'www.linkedin.com',
    'quora.com', 'www.quora.com',
    'archive.ph', 'archive.is', 'archive.vn', 'archive.today',
    'web.archive.org',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fill empty url2text fields in knowledge store files"
    )
    parser.add_argument(
        '--input_dir', type=str,
        default='dataset/AVerImaTeC_Shared_Task/Knowledge_Store/val/text_related/image_related_store_text_val',
        help='Input directory with JSON files'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default='dataset/AVerImaTeC_Shared_Task/Knowledge_Store/val/text_related/image_related_store_text_val_filled',
        help='Output directory for filled JSON files'
    )
    parser.add_argument(
        '--timeout', type=int,
        default=20,
        help='Timeout for URL requests in seconds'
    )
    parser.add_argument(
        '--delay', type=float,
        default=2.0,
        help='Delay between requests in seconds'
    )
    parser.add_argument(
        '--start_idx', type=int,
        default=1,
        help='Start file index (1-indexed)'
    )
    parser.add_argument(
        '--end_idx', type=int,
        default=-1,
        help='End file index (-1 for all)'
    )
    parser.add_argument(
        '--use_playwright', action='store_true',
        default=True,
        help='Use Playwright for JavaScript-heavy sites'
    )
    return parser.parse_args()


def get_domain(url: str) -> str:
    """Extract domain from URL."""
    if '://' in url:
        domain = url.split('://')[1].split('/')[0]
        return domain.lower()
    return ''


def needs_playwright(url: str) -> bool:
    """Check if URL needs Playwright for scraping."""
    domain = get_domain(url)
    # Check exact match or if domain ends with any playwright domain
    for pw_domain in PLAYWRIGHT_DOMAINS:
        if domain == pw_domain or domain.endswith('.' + pw_domain):
            return True
    return False


def create_playwright_browser(timeout: int = 20):
    """Create a Playwright browser instance."""
    from playwright.sync_api import sync_playwright

    playwright = sync_playwright().start()

    # Try Chromium first
    try:
        browser = playwright.chromium.launch(
            headless=True,
            args=[
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-gpu',
            ]
        )
        context = browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        context.set_default_timeout(timeout * 1000)  # Playwright uses ms
        print("Using Playwright Chromium")
        return playwright, browser, context
    except Exception as e:
        print(f"Chromium failed: {e}")
        playwright.stop()
        raise RuntimeError("Failed to launch Playwright browser")


def extract_text_from_html(html_content: str) -> List[str]:
    """Extract text from HTML content using BeautifulSoup."""
    from bs4 import BeautifulSoup
    import re

    soup = BeautifulSoup(html_content, 'html.parser')

    texts = []

    # First, try to extract from meta tags (useful for social media login walls)
    meta_texts = []
    for meta in soup.find_all('meta'):
        prop = meta.get('property', '') or meta.get('name', '')
        content = meta.get('content', '')
        if prop in ['og:title', 'og:description', 'twitter:title', 'twitter:description', 'description']:
            if content and len(content) > 30:
                meta_texts.append(content.strip())

    # Extract from JSON-LD structured data (common in YouTube, news sites)
    json_ld_texts = []
    for script in soup.find_all('script', type='application/ld+json'):
        try:
            data = json.loads(script.string or '')
            # Handle both single objects and arrays
            items = data if isinstance(data, list) else [data]
            for item in items:
                # Extract description from VideoObject, Article, etc.
                if isinstance(item, dict):
                    for key in ['description', 'articleBody', 'text']:
                        if key in item and isinstance(item[key], str) and len(item[key]) > 50:
                            json_ld_texts.append(item[key].strip())
                    # Also check name/headline
                    for key in ['name', 'headline']:
                        if key in item and isinstance(item[key], str) and len(item[key]) > 20:
                            json_ld_texts.append(item[key].strip())
        except (json.JSONDecodeError, TypeError):
            continue

    # YouTube-specific: Extract from ytInitialData or ytInitialPlayerResponse
    for script in soup.find_all('script'):
        script_text = script.string or ''
        # Look for YouTube's initial data
        for var_name in ['ytInitialData', 'ytInitialPlayerResponse']:
            match = re.search(rf'var {var_name}\s*=\s*(\{{.*?\}});', script_text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                    # Try to extract video description
                    desc = None
                    # Path for ytInitialPlayerResponse
                    try:
                        desc = data.get('videoDetails', {}).get('shortDescription', '')
                    except (AttributeError, TypeError):
                        pass
                    if desc and len(desc) > 50:
                        json_ld_texts.append(desc.strip())
                except (json.JSONDecodeError, TypeError):
                    continue

    # Remove unwanted elements
    for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        element.decompose()

    # Try to get main content first
    main_content = soup.find(['main', 'article']) or soup.find('div', {'class': ['content', 'article', 'post']})
    if main_content:
        paragraphs = main_content.find_all(['p', 'h1', 'h2', 'h3', 'li', 'span', 'div'])
    else:
        paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'li', 'span', 'div'])

    for p in paragraphs:
        text = p.get_text(separator=' ', strip=True)
        # Filter out short or empty text
        if len(text) > 30:
            # Clean up whitespace
            text = ' '.join(text.split())
            texts.append(text)

    # Filter out login wall / blocked content indicators
    login_indicators = [
        'log in', 'login', 'sign in', 'sign up', 'create account',
        '로그인', '계정을 잊으셨나요', 'forgot account', 'forgot password',
        'you must log in', 'please log in', 'create new account',
        'logged in to comment', 'logged in to reply'
    ]
    filtered_texts = []
    for text in texts:
        text_lower = text.lower()
        # Skip if text is primarily login-related
        if any(indicator in text_lower for indicator in login_indicators):
            # Only skip if it's short (likely a button/link) or mostly login content
            if len(text) < 100:
                continue
        filtered_texts.append(text)

    # Combine JSON-LD texts (full content), meta texts, then body texts
    # JSON-LD often has the full description (e.g., YouTube video descriptions)
    all_texts = json_ld_texts + meta_texts + filtered_texts

    # Deduplicate while preserving order, also filter out substrings
    seen = set()
    unique_texts = []
    for text in all_texts:
        if text not in seen:
            # Check if this text is a substring of any already added text
            is_substring = any(text in existing for existing in unique_texts)
            if not is_substring:
                seen.add(text)
                unique_texts.append(text)

    return unique_texts[:30]  # Limit to 30 text blocks


def fetch_url_text_requests(url: str, timeout: int = 15) -> List[str]:
    """Fetch and extract text content from a URL using requests."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }

        response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        response.raise_for_status()

        # Check content type
        content_type = response.headers.get('content-type', '')
        if 'text/html' not in content_type and 'text/plain' not in content_type:
            return []

        return extract_text_from_html(response.content)

    except Exception:
        return []


def fetch_url_text_playwright(context: Any, url: str) -> List[str]:
    """Fetch and extract text content from a URL using Playwright."""
    try:
        page = context.new_page()
        try:
            page.goto(url, wait_until='domcontentloaded')

            # Wait for page to load
            time.sleep(2)  # Basic wait for JavaScript to render

            # Scroll down to load lazy content
            page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
            time.sleep(1)

            # Get page source after JavaScript rendering
            html_content = page.content()

            return extract_text_from_html(html_content)
        finally:
            page.close()

    except Exception:
        return []


def fetch_url_text(url: str, timeout: int = 20, context: Any = None) -> List[str]:
    """Fetch and extract text content from a URL. Uses Playwright for JS-heavy sites."""
    if needs_playwright(url) and context is not None:
        return fetch_url_text_playwright(context, url)
    else:
        # Try requests first
        result = fetch_url_text_requests(url, timeout)
        # If requests fails and we have a context, try Playwright
        if not result and context is not None:
            result = fetch_url_text_playwright(context, url)
        return result


def has_useful_content(url2text: List[str], min_total_length: int = 100) -> bool:
    """
    Check if url2text contains useful information or just generic/login content.

    Returns False if:
    - The merged text is too short
    - The content is mostly generic site elements (footer, navigation)
    - The content is login wall text
    """
    if not url2text:
        return False

    # Merge all texts
    merged_text = ' '.join(url2text).lower()

    # Check minimum total length
    if len(merged_text) < min_total_length:
        return False

    # Generic site elements that indicate no real content was captured
    generic_patterns = [
        # YouTube generic footer
        'about', 'press', 'copyright', 'contact us', 'creators', 'advertise',
        'developers', 'terms', 'privacy', 'policy & safety', 'how youtube works',
        'test new features', 'nfl sunday ticket', '© 2025 google llc',
        # Common footer/nav elements
        'cookie policy', 'cookie settings', 'privacy policy', 'terms of service',
        'terms of use', 'all rights reserved', 'sitemap',
        # Login wall indicators (multiple languages)
        '로그인', '계정을 잊으셨나요', 'log in', 'sign in', 'sign up',
        'create account', 'forgot password', 'forgot account',
        'you must log in', 'please log in', 'create new account',
        'logged in to comment', 'logged in to reply',
        # Social media generic
        'follow us', 'share this', 'like us on facebook',
    ]

    # Count how many generic patterns are found
    generic_count = sum(1 for pattern in generic_patterns if pattern in merged_text)

    # If more than 30% of the content words match generic patterns, it's not useful
    words_in_merged = len(merged_text.split())
    if words_in_merged > 0 and generic_count >= 5:
        # Check if most of the content is just generic elements
        # by seeing if the content is dominated by short generic phrases
        non_generic_length = len(merged_text)
        for pattern in generic_patterns:
            non_generic_length -= merged_text.count(pattern) * len(pattern)

        # If less than 50% of content is non-generic, it's not useful
        if non_generic_length < len(merged_text) * 0.5:
            return False

    # Check if it's primarily login wall content
    login_keywords = ['로그인', 'log in', 'sign in', 'sign up', 'create account']
    login_count = sum(1 for kw in login_keywords if kw in merged_text)
    if login_count >= 2 and len(merged_text) < 500:
        return False

    return True


def process_file(input_path: str, output_path: str, timeout: int, delay: float,
                 context: Any = None) -> dict:
    """Process a single JSON file, filling empty or useless url2text fields."""
    stats = {'total': 0, 'filled': 0, 'already_filled': 0, 'failed': 0, 'refilled': 0}

    # Read input file (JSON Lines format)
    entries = []
    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    stats['total'] = len(entries)

    # Process each entry
    filled_entries = []
    for entry in entries:
        url2text = entry.get('url2text', [])

        if url2text and has_useful_content(url2text):
            # Already has useful content
            stats['already_filled'] += 1
            filled_entries.append(entry)
        else:
            # Need to fetch (empty or useless content)
            needs_refill = bool(url2text)  # Track if this is a refill
            url = entry.get('url', '')
            if url:
                fetched_text = fetch_url_text(url, timeout, context)
                if fetched_text and has_useful_content(fetched_text):
                    entry['url2text'] = fetched_text
                    if needs_refill:
                        stats['refilled'] += 1
                    else:
                        stats['filled'] += 1
                else:
                    # Keep original if refetch didn't get better content
                    if needs_refill and url2text:
                        stats['already_filled'] += 1
                    else:
                        stats['failed'] += 1
                time.sleep(delay)  # Rate limiting
            filled_entries.append(entry)

    # Write output file
    with open(output_path, 'w') as f:
        for entry in filled_entries:
            f.write(json.dumps(entry) + '\n')

    return stats


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get list of input files
    input_files = sorted([
        f for f in os.listdir(args.input_dir)
        if f.endswith('.json')
    ], key=lambda x: int(x.replace('.json', '')))

    # Determine range to process
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx > 0 else len(input_files) + 1

    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Processing files {start_idx} to {end_idx - 1}")
    print(f"Timeout: {args.timeout}s, Delay: {args.delay}s")
    print(f"Playwright enabled: {args.use_playwright}")
    print()

    # Create Playwright browser if enabled
    playwright = None
    browser = None
    context = None
    if args.use_playwright:
        try:
            print("Initializing Playwright browser...")
            playwright, browser, context = create_playwright_browser(args.timeout)
            print("Playwright browser initialized successfully.")
        except Exception as e:
            print(f"Warning: Failed to initialize Playwright: {e}")
            print("Falling back to requests-only mode.")
            playwright = browser = context = None

    # Aggregate stats
    total_stats = {'total': 0, 'filled': 0, 'already_filled': 0, 'failed': 0, 'refilled': 0}

    try:
        # Process files
        for filename in tqdm(input_files, desc="Processing files"):
            file_idx = int(filename.replace('.json', ''))

            # Skip if outside range
            if file_idx < start_idx or file_idx >= end_idx:
                continue

            input_path = os.path.join(args.input_dir, filename)
            output_path = os.path.join(args.output_dir, filename)

            # Skip if output already exists
            if os.path.exists(output_path):
                continue

            # Process file
            stats = process_file(input_path, output_path, args.timeout, args.delay, context)

            # Update aggregate stats
            for key in total_stats:
                total_stats[key] += stats[key]

    finally:
        # Clean up Playwright
        if browser is not None:
            try:
                browser.close()
            except Exception:
                pass
        if playwright is not None:
            try:
                playwright.stop()
                print("\nPlaywright browser closed.")
            except Exception:
                pass

    # Print summary
    print(f"\n{'='*50}")
    print("Summary:")
    print(f"  Total entries: {total_stats['total']}")
    print(f"  Already filled: {total_stats['already_filled']}")
    print(f"  Newly filled: {total_stats['filled']}")
    print(f"  Re-filled (had useless content): {total_stats['refilled']}")
    print(f"  Failed to fetch: {total_stats['failed']}")
    print(f"\nOutput saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

