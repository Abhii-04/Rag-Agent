import os
from typing import Literal

from dotenv import load_dotenv
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from playwright.async_api import async_playwright
from tavily import TavilyClient

from tools.rag import rag_search, save_to_knowledge

load_dotenv(override=True)


def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news"] = "general",
    include_raw_content: bool = False,
):
    """Search the internet for recent/latest information."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set in the environment or .env file.")

    tavily = TavilyClient(api_key)
    return tavily.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


async def playwright_tools():
    """Create Playwright browser tools for advanced browser automation."""
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(
        channel="chrome",
        headless=False,
    )
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
    return toolkit.get_tools(), playwright, browser


def build_wiki_tool():
    """Build the Wikipedia LangChain tool."""
    wikipedia = WikipediaAPIWrapper()
    return WikipediaQueryRun(api_wrapper=wikipedia)


__all__ = [
    "build_wiki_tool",
    "internet_search",
    "playwright_tools",
    "rag_search",
    "save_to_knowledge",
]
