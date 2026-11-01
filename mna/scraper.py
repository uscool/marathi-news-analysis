"""Step 1: open a news URL in a real browser (Selenium, or Playwright) and
extract the Marathi article body."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

from .text_utils import devanagari_ratio, language_check, normalize

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

# Lines that appear inside article containers on Marathi news sites but are not news text.
BOILERPLATE = re.compile(
    r"(हेही वाचा|हे पण वाचा|अधिक वाचा|आणखी वाचा|ताज्या बातम्या|जाहिरात|सबस्क्राईब|"
    r"फॉलो करा|डाउनलोड करा|लोकप्रिय बातम्या|संबंधित बातम्या|Advertisement|Follow us|"
    r"Read more|Also read|Subscribe|Download)", re.I)


class NotMarathiError(ValueError):
    pass


@dataclass
class Article:
    url: str
    title: str
    text: str
    paragraphs: list[str] = field(default_factory=list)
    method: str = ""
    language: dict = field(default_factory=dict)


# --------------------------------------------------------------------- browsers
def fetch_html_selenium(url: str, timeout: int = 30, headless: bool = True) -> str:
    from selenium import webdriver
    from selenium.common.exceptions import TimeoutException
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    opts = webdriver.ChromeOptions()
    if headless:
        opts.add_argument("--headless=new")
    for arg in ("--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu",
                "--lang=mr-IN", f"--user-agent={USER_AGENT}", "--window-size=1366,2000"):
        opts.add_argument(arg)
    opts.page_load_strategy = "eager"          # don't wait for every ad/tracker
    driver = webdriver.Chrome(options=opts)    # Selenium Manager fetches the driver
    try:
        driver.set_page_load_timeout(timeout)
        try:
            driver.get(url)
        except TimeoutException:
            pass                               # use whatever has loaded
        WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.TAG_NAME, "p")))
        for _ in range(4):                     # trigger lazy-loaded paragraphs
            driver.execute_script("window.scrollBy(0, document.body.scrollHeight / 4);")
            time.sleep(0.6)
        return driver.page_source
    finally:
        driver.quit()


def fetch_html_playwright(url: str, timeout: int = 30, headless: bool = True) -> str:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        try:
            page = browser.new_page(user_agent=USER_AGENT, locale="mr-IN")
            page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
            page.wait_for_selector("p", timeout=timeout * 1000)
            for _ in range(4):
                page.mouse.wheel(0, 1500)
                page.wait_for_timeout(600)
            return page.content()
        finally:
            browser.close()


# -------------------------------------------------------------------- extraction
def _clean_paragraphs(paras: list[str]) -> list[str]:
    seen, out = set(), []
    for p in paras:
        p = normalize(p)
        if len(p) < 30 or devanagari_ratio(p) < 0.5 or BOILERPLATE.search(p):
            continue
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _title(soup: BeautifulSoup) -> str:
    for sel in ('meta[property="og:title"]', 'meta[name="twitter:title"]'):
        tag = soup.select_one(sel)
        if tag and tag.get("content"):
            return normalize(tag["content"])
    h1 = soup.find("h1")
    if h1:
        return normalize(h1.get_text(" "))
    return normalize(soup.title.get_text()) if soup.title else ""


def _heuristic_body(soup: BeautifulSoup) -> list[str]:
    """Pick the container whose direct <p> children hold the most Devanagari text."""
    for tag in soup(["script", "style", "noscript", "nav", "footer", "aside", "header", "form", "iframe"]):
        tag.decompose()
    best, best_score = None, 0
    for parent in soup.find_all(["article", "div", "section", "main"]):
        ps = parent.find_all("p", recursive=False)
        score = sum(len(re.findall(r"[\u0900-\u097F]", p.get_text())) for p in ps)
        if score > best_score:
            best, best_score = parent, score
    if best is not None and best_score > 300:
        return [p.get_text(" ") for p in best.find_all("p", recursive=False)]
    return [p.get_text(" ") for p in soup.find_all("p")]


def extract_article(html: str, url: str = "") -> tuple[str, list[str], str]:
    soup = BeautifulSoup(html, "html.parser")
    title = _title(soup)
    paras, method = [], ""
    try:
        import trafilatura
        txt = trafilatura.extract(html, include_comments=False, include_tables=False, favor_recall=True)
        if txt:
            paras, method = _clean_paragraphs(txt.split("\n")), "trafilatura"
    except ImportError:
        pass
    if sum(len(p) for p in paras) < 400:  # trafilatura missed it -> DOM heuristic
        paras, method = _clean_paragraphs(_heuristic_body(soup)), "dom-heuristic"
    return title, paras, method


def scrape(url: str, engine: str = "selenium", timeout: int = 30) -> Article:
    fetch = fetch_html_playwright if engine == "playwright" else fetch_html_selenium
    html = fetch(url, timeout=timeout)
    title, paras, method = extract_article(html, url)
    text = "\n".join(paras)
    lang = language_check(text)
    if not paras:
        raise ValueError("No article text found on this page. Paste the text instead.")
    if not lang["is_marathi"]:
        raise NotMarathiError(f"This page does not look like Marathi news: {lang}")
    return Article(url=url, title=title, text=text, paragraphs=paras,
                   method=f"{engine}+{method}", language=lang)
