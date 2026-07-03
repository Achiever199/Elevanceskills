"""
Entrypoint for the dynamic knowledge-base system.

Usage:
    python main.py serve                 # run scheduler as a long-lived service
    python main.py update-now            # run a single update pass for all sources then exit
    python main.py update-now --source docs_folder   # update just one source
    python main.py chat                  # interactive REPL against the knowledge base
    python main.py stats                 # print vector store / manifest stats
"""
from __future__ import annotations

import argparse
import signal
import sys
import time

from dotenv import load_dotenv

from src.chatbot import KnowledgeBaseChatbot
from src.config import load_config
from src.manifest import Manifest
from src.scheduler import KnowledgeBaseScheduler
from src.updater import KnowledgeBaseUpdater
from src.utils import setup_logging
from src.vector_store import VectorStore

load_dotenv()


def build_components(config_path: str):
    config = load_config(config_path)
    log_cfg = config.logging_cfg
    logger = setup_logging(level=log_cfg.get("level", "INFO"), log_file=log_cfg.get("file", "./logs/kb_system.log"))
    vs_cfg = config.vector_store
    vector_store = VectorStore(
        persist_directory=vs_cfg.get("persist_directory", "./data/chroma"),
        collection_name=vs_cfg.get("collection_name", "knowledge_base"),
        embedding_model=vs_cfg.get("embedding_model", "all-MiniLM-L6-v2"),
        embedding_provider=vs_cfg.get("embedding_provider", "default"),
    )
    manifest = Manifest(config.manifest_path)
    updater = KnowledgeBaseUpdater(config, vector_store, manifest)
    return config, logger, vector_store, manifest, updater


def cmd_serve(args):
    config, logger, vector_store, manifest, updater = build_components(args.config)
    scheduler = KnowledgeBaseScheduler(config, updater)
    scheduler.start(run_immediately=not args.skip_initial_run)

    logger.info("Knowledge base service is running. Press Ctrl+C to stop.")

    stop = {"flag": False}

    def _handle_signal(signum, frame):  # noqa: ARG001
        logger.info("Shutdown signal received.")
        stop["flag"] = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        while not stop["flag"]:
            time.sleep(1)
    finally:
        scheduler.shutdown()
        logger.info("Service stopped.")


def cmd_update_now(args):
    config, logger, vector_store, manifest, updater = build_components(args.config)
    if args.source:
        matching = [s for s in config.sources if s["id"] == args.source]
        if not matching:
            print(f"No enabled source with id '{args.source}' found in config.")
            sys.exit(1)
        results = [updater.update_source(matching[0])]
    else:
        results = updater.update_all()

    print("\nUpdate summary:")
    for r in results:
        print(
            f"  - {r['source_id']}: new={r['new']} updated={r['updated']} "
            f"unchanged={r['unchanged']} removed={r['removed']} errors={r['errors']}"
        )
    print(f"\nTotal vectors in store: {vector_store.count()}")


def cmd_chat(args):
    config, logger, vector_store, manifest, updater = build_components(args.config)
    bot = KnowledgeBaseChatbot(vector_store)
    print("Knowledge-base chat. Type 'exit' to quit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        result = bot.ask(question)
        print(f"\nBot: {result['answer']}\n")


def cmd_stats(args):
    config, logger, vector_store, manifest, updater = build_components(args.config)
    print(f"Vector count: {vector_store.count()}")
    for source_cfg in config.sources:
        doc_ids = manifest.all_doc_ids_for_source(source_cfg["id"])
        print(f"  - {source_cfg['id']}: {len(doc_ids)} tracked document(s)")


def main():
    parser = argparse.ArgumentParser(description="Dynamic knowledge-base update system")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="Run the scheduler as a long-lived service")
    p_serve.add_argument("--skip-initial-run", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    p_update = sub.add_parser("update-now", help="Run one update pass and exit")
    p_update.add_argument("--source", help="Only update this source id")
    p_update.set_defaults(func=cmd_update_now)

    p_chat = sub.add_parser("chat", help="Interactive chat against the knowledge base")
    p_chat.set_defaults(func=cmd_chat)

    p_stats = sub.add_parser("stats", help="Print vector store / manifest stats")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
