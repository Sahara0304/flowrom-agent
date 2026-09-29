from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import requests


API_URL = (
    "https://api.semanticscholar.org/graph/v1/paper/search"
)


class LiteratureSearchError(RuntimeError):
    """文献搜索相关异常。"""

    pass


def search_papers(
    query: str,
    limit: int = 6,
    year_start: int = 2020,
    year_end: int = 2026,
    max_retries: int = 3,
) -> list[dict[str, Any]]:
    """
    使用 Semantic Scholar 搜索论文。

    主要特性：
    - 支持 API Key
    - HTTP 429 自动退避
    - 网络错误自动重试
    - 最大重试次数有限
    - 不会无限访问 API
    """

    params = {
        "query": query,
        "limit": limit,
        "fields": (
            "paperId,"
            "title,"
            "abstract,"
            "year,"
            "authors,"
            "url,"
            "citationCount,"
            "openAccessPdf"
        ),
        "year": f"{year_start}-{year_end}",
        "sort": "publicationDate:desc",
    }

    headers = {}

    api_key = os.getenv(
        "SEMANTIC_SCHOLAR_API_KEY"
    )

    if api_key:
        headers["x-api-key"] = api_key

    for attempt in range(
        max_retries + 1
    ):

        # ====================================================
        # HTTP 请求
        # ====================================================

        try:

            response = requests.get(
                API_URL,
                params=params,
                headers=headers,
                timeout=30,
            )

        except requests.RequestException as exc:

            if attempt >= max_retries:

                raise LiteratureSearchError(
                    f"Network error after "
                    f"{max_retries + 1} attempts: "
                    f"{exc}"
                ) from exc

            sleep_time = min(
                2 ** attempt,
                10,
            )

            print(
                f"[Literature] "
                f"network error: {exc}"
            )

            print(
                f"[Literature] "
                f"retrying in "
                f"{sleep_time}s..."
            )

            time.sleep(
                sleep_time
            )

            continue

        # ====================================================
        # 429 Rate Limit
        # ====================================================

        if response.status_code == 429:

            if attempt >= max_retries:

                raise LiteratureSearchError(
                    "Semantic Scholar API "
                    "rate limit after retries."
                )

            retry_after = (
                response.headers.get(
                    "Retry-After"
                )
            )

            if retry_after:

                try:
                    sleep_time = float(
                        retry_after
                    )

                except ValueError:
                    sleep_time = (
                        2 ** attempt
                    )

            else:
                sleep_time = (
                    2 ** attempt
                )

            # 至少等待 1.5 秒
            sleep_time = max(
                sleep_time,
                1.5,
            )

            print(
                "[Literature] "
                f"HTTP 429."
            )

            print(
                "[Literature] "
                f"waiting {sleep_time:.1f}s..."
            )

            time.sleep(
                sleep_time
            )

            continue

        # ====================================================
        # Other HTTP errors
        # ====================================================

        if response.status_code != 200:

            raise LiteratureSearchError(
                "Semantic Scholar API error: "
                f"HTTP {response.status_code}\n"
                f"{response.text[:1000]}"
            )

        # ====================================================
        # JSON
        # ====================================================

        try:

            data = response.json()

        except ValueError as exc:

            raise LiteratureSearchError(
                "Semantic Scholar returned "
                "invalid JSON."
            ) from exc

        papers = []

        for paper in data.get(
            "data",
            [],
        ):

            authors = []

            for author in paper.get(
                "authors",
                [],
            ):

                name = author.get(
                    "name"
                )

                if name:
                    authors.append(name)

            open_access_pdf = (
                paper.get(
                    "openAccessPdf"
                )
            )

            papers.append(
                {
                    "paper_id": (
                        paper.get(
                            "paperId"
                        )
                    ),
                    "title": (
                        paper.get(
                            "title"
                        )
                    ),
                    "abstract": (
                        paper.get(
                            "abstract"
                        )
                    ),
                    "year": (
                        paper.get(
                            "year"
                        )
                    ),
                    "authors": authors,
                    "url": (
                        paper.get(
                            "url"
                        )
                    ),
                    "citation_count": (
                        paper.get(
                            "citationCount"
                        )
                    ),
                    "open_access_pdf": (
                        open_access_pdf
                    ),
                }
            )

        # ====================================================
        # 正常情况下限制请求频率
        # ====================================================

        time.sleep(1.5)

        return papers

    raise LiteratureSearchError(
        "Unknown literature search error."
    )


def search_multiple_queries(
    queries: list[str],
    limit_each: int = 6,
) -> dict[str, list[dict[str, Any]]]:
    """
    顺序执行多个 query。

    如果确认遇到 429：
        后面的 query 不再继续发送。
    """

    results = {}

    rate_limited = False

    for index, query in enumerate(
        queries,
        start=1,
    ):

        print()
        print(
            f"[Literature] "
            f"Query {index}/{len(queries)}:"
        )

        print(
            f"  {query}"
        )

        try:

            papers = search_papers(
                query=query,
                limit=limit_each,
            )

            results[query] = papers

            print(
                f"[Literature] "
                f"Retrieved {len(papers)} papers."
            )

        except LiteratureSearchError as exc:

            print(
                "[Literature] "
                f"failed: {exc}"
            )

            results[query] = []

            error_text = str(
                exc
            ).lower()

            if (
                "rate limit"
                in error_text
                or "429"
                in error_text
            ):

                rate_limited = True

        if rate_limited:

            print()
            print(
                "[Literature] "
                "Stopping further queries "
                "because rate limit was detected."
            )

            break

    return results