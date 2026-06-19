import argparse
import os
import uuid
from datetime import date
from pathlib import Path

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from prompts import research_instructions
from tools.tool import build_wiki_tool, internet_search, rag_search, save_to_knowledge

load_dotenv(override=True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_model() -> ChatOpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set in the environment or .env file.")

    return ChatOpenAI(
        api_key=api_key,
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )


def build_tools():
    return [rag_search, save_to_knowledge, internet_search, build_wiki_tool()]


def build_agent():
    llm = build_model()
    tools = build_tools()

    return create_deep_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a professional research orchestrator. Break complex requests into subtasks, "
            "use local RAG before web search when the topic may exist in the knowledge base, "
            "delegate research to the research_agent when fresh information is needed, "
            "use the evaluator for quality checks, and return polished, source-aware results."
        ),
        permissions=[
            FilesystemPermission(
                operations=["read", "write"],
                paths=["/**"],
                mode="allow",
            )
        ],
        backend=FilesystemBackend(root_dir=PROJECT_ROOT, virtual_mode=True),
        memory=["/memory/AGENT.md"],
        skills=["/skills"],
        subagents=[
            {
                "name": "research_agent",
                "description": "Use this agent for research tasks and gathering information.",
                "system_prompt": research_instructions,
                "tools": tools,
            },
            {
                "name": "evaluator",
                "description": "Use this agent to evaluate whether output meets the required criteria.",
                "system_prompt": (
                    "You evaluate whether the work performed by the agents satisfies the user request. "
                    "Check accuracy, completeness, formatting, and whether evidence is included when required."
                ),
                "tools": [],
            },
        ],
    )


def run_research(topic: str, output_path: str = "/research_report.txt"):
    orchestrator = build_agent()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    current_date = date.today().isoformat()
    prompt = (
        f"Research this topic professionally: {topic}\n\n"
        f"Today's date is {current_date}. Use this exact date for report metadata and for interpreting latest/current claims. "
        "Use local RAG first for relevant stored context. Then use web search for fresh facts, "
        "verification, and sources. Produce a structured research report with an executive summary, "
        "key findings, source notes, and uncertainty or limitations. "
        f"Write the report to {output_path}, then tell me when it is done."
    )
    return orchestrator.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config=config,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DeepAgent professional research agent.")
    parser.add_argument(
        "topic",
        nargs="?",
        default="latest news on Agentic AI development",
        help="Research topic to investigate.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="/research_report.txt",
        help="Virtual output path for the generated report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_research(args.topic, args.output)
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
