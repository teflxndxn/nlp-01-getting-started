"""
web_words_case.py - Project script (example).

Purpose

  Retrieve a web page, extract its text, count word frequencies,
  and visualize the results.

Analytical Questions

- What terms dominate this page?
- What topic does the vocabulary suggest?
- What noise appears from navigation or markup?
- What preprocessing steps might improve results?

Notes

- This example performs simple frequency analysis.
- More advanced text processing techniques will be introduced later.

Run from root project folder with:

  python src/nlp/web_words_case.py
"""

import logging
from pathlib import Path

from bs4 import BeautifulSoup
from datafun_toolkit.logger import get_logger, log_header
import matplotlib.pyplot as plt
import polars as pl
import requests
from wordcloud import WordCloud


def main() -> None:
    print("Imports complete.")

    log: logging.Logger = get_logger("CI", level="DEBUG")

    root_path: Path = Path.cwd()
    notebooks_path: Path = root_path / "notebooks"
    scripts_path: Path = root_path / "scripts"

    log_header(log, "NLP")
    log.info("START script.....")

    log.info(f"ROOT_PATH: {root_path}")
    log.info(f"NOTEBOOKS_PATH: {notebooks_path}")
    log.info(f"SCRIPTS_PATH: {scripts_path}")

    url: str = "https://en.wikipedia.org/wiki/Natural_language_processing"
    headers: dict[str, str] = {
        "User-Agent": "Mozilla/5.0 (compatible; NLP-Course-Example/1.0)"
    }

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    html: str = response.text

    print(f"Downloaded {len(html):,} characters from:")
    print(url)

    try:
        soup: BeautifulSoup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    print("HTML parsed successfully.")
    print(type(soup))

    text: str = soup.get_text(separator=" ", strip=True)

    print("First 1000 characters of extracted text:")
    print(text[:1000])

    words: list[str] = text.split()
    print("First 20 raw words:")
    print(words[:20])
    print(f"Total raw words: {len(words):,}")

    words = [word.lower() for word in words]

    clean_words: list[str] = [
        word.strip(".,:;!?()[]\"'") for word in words if len(word) > 3
    ]

    print("First 20 cleaned words:")
    print(clean_words[:20])
    print(f"Total cleaned words: {len(clean_words):,}")

    df: pl.DataFrame = pl.DataFrame({"word": clean_words})
    freq_df: pl.DataFrame = df.group_by("word").len().sort("len", descending=True)

    print("Top 20 most frequent words:")
    print(freq_df.head(20))

    top_df: pl.DataFrame = freq_df.head(10)

    plt.figure(figsize=(10, 5))
    plt.bar(top_df["word"], top_df["len"])
    ax = plt.gca()
    ax.tick_params(axis="x", labelrotation=45)
    plt.title("Most Frequent Words")
    plt.xlabel("Word")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    freq_dict: dict[str, int] = dict(
        zip(freq_df["word"].to_list(), freq_df["len"].to_list(), strict=True)
    )

    print("Sample of word frequencies:")
    for word, freq in list(freq_dict.items())[:10]:
        print(f"{word}: {freq}")

    wc: WordCloud = WordCloud(width=1000, height=500, background_color="white")
    wc.generate_from_frequencies(freq_dict)

    plt.figure(figsize=(12, 6))
    plt.imshow(wc)
    plt.axis("off")
    plt.title("Word Cloud")
    plt.show()

    log.info("========================")
    log.info("Pipeline executed successfully!")
    log.info("========================")
    log.info("END main()")

    project_log = root_path / "project.log"

    with project_log.open("w", encoding="utf-8") as file:
        file.write("========================\n")
        file.write("Pipeline executed successfully!\n")
        file.write("========================\n")

    print("========================")
    print("Pipeline executed successfully!")
    print("========================")


if __name__ == "__main__":
    main()
