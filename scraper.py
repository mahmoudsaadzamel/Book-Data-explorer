from __future__ import annotations
import csv
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, List, Dict
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE = "https://books.toscrape.com/"
HEADERS = {"User-Agent": "books-scraper/1.0 (+https://example.com)"}

# only scrape the 4 categories used in the task
TARGET_CATEGORIES = {"Classics", "Historical Fiction", "Mystery", "Travel"}

RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

@dataclass
class Book:
    title: str
    category: str
    price: float
    availability_text: str
    availability_n: int
    rating: int
    description: str
    upc: str
    product_url: str
    image_url: str
    in_stock: bool
    desc_word_count: int

def get_soup(url: str) -> BeautifulSoup:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "lxml")  # html.parser also fine

def categories() -> Dict[str, str]:
    """Return mapping: Category Name -> URL"""
    soup = get_soup(BASE)
    items = soup.select("div.side_categories ul li ul li a")
    mapping = {}
    for a in items:
        name = a.get_text(strip=True)
        href = a.get("href")
        mapping[name] = urljoin(BASE, href)
    return mapping

def paginate_category(first_page_url: str) -> Iterator[str]:
    """Yield all page URLs in a category by following 'next'."""
    url = first_page_url
    while True:
        yield url
        soup = get_soup(url)
        nxt = soup.select_one("li.next a")
        if not nxt:
            break
        url = urljoin(url, nxt.get("href"))

def product_links_from_listing(listing_url: str) -> List[str]:
    soup = get_soup(listing_url)
    links = []
    for a in soup.select("article.product_pod h3 a"):
        href = a.get("href")
       
        abs_url = urljoin(listing_url, href)
        if "/catalogue/" not in abs_url:
            abs_url = urljoin(BASE, f"catalogue/{href}")
        links.append(abs_url)
    return links

def parse_book_page(url: str, category: str) -> Book:
    soup = get_soup(url)

    title = soup.select_one("div.product_main h1").get_text(strip=True)

    price_text = soup.select_one("p.price_color").get_text(strip=True)
    price = float(re.sub(r"[^0-9.]", "", price_text))

    avail_el = soup.select_one("p.instock.availability")
    availability_text = avail_el.get_text(strip=True) if avail_el else ""
    m = re.search(r"(\d+)", availability_text)
    availability_n = int(m.group(1)) if m else 0

    rating_el = soup.select_one("p.star-rating")
    rating_word = next((cls for cls in rating_el.get("class", []) if cls in RATING_MAP), "One")
    rating = RATING_MAP[rating_word]

   
    desc = ""
    hdr = soup.select_one("#product_description")
    if hdr:
        p = hdr.find_next("p")
        if p:
            desc = p.get_text(" ", strip=True)
    desc_word_count = len(desc.split())

   
    upc = ""
    for row in soup.select("table.table.table-striped tr"):
        th = row.select_one("th")
        td = row.select_one("td")
        if th and th.get_text(strip=True) == "UPC" and td:
            upc = td.get_text(strip=True)
            break

    img = soup.select_one("div.item.active img")
    image_url = urljoin(url, img.get("src")) if img else ""

    in_stock = availability_n > 0

    return Book(
        title=title,
        category=category,
        price=price,
        availability_text=availability_text,
        availability_n=availability_n,
        rating=rating,
        description=desc,
        upc=upc,
        product_url=url,
        image_url=image_url,
        in_stock=in_stock,
        desc_word_count=desc_word_count,
    )

def scrape(categories_whitelist: set[str] = TARGET_CATEGORIES, pause: float = 0.3) -> List[Book]:
    cat_map = categories()
    books: List[Book] = []
    for cat_name, cat_url in cat_map.items():
        if cat_name not in categories_whitelist:
            continue
        for page_url in paginate_category(cat_url):
            for prod_url in product_links_from_listing(page_url):
                try:
                    books.append(parse_book_page(prod_url, category=cat_name))
                except Exception as e:
                    print(f"[warn] Failed {prod_url}: {e}")
                time.sleep(pause)  # be polite
    return books

def save_csv(path: str | Path, books: List[Book]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(b) for b in books]
    cols = [
        "title","category","price","availability_text","availability_n",
        "rating","description","upc","product_url","image_url","in_stock",
        "desc_word_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

if __name__ == "__main__":
    out = Path("books.csv")
    print("[info] scraping…")
    data = scrape()
    print(f"[info] scraped {len(data)} books")
    save_csv(out, data)
    print(f"[info] saved to {out.resolve()}")
