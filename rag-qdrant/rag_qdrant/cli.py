"""Thin, ultra-fast command-line interface communicating with the persistent rag-qdrant background service."""

import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure unbuffered UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

from .arguments import create_parser, create_search_parser, validate_indexing_arguments
from .client import RagServiceClient, ServiceError, ServiceUnavailableError
from .config import (
    COLLECTION_NAME,
    NUM_WORKERS,
    QDRANT_URL,
    SERVICE_HOST,
    SERVICE_PORT,
    SERVICE_URL,
)

HELP_TEXT = """
rag_qdrant — lokalny, przyrostowy indeksator Markdown dla Qdrant

CEL
  Indeksuje dokumenty Markdown do kolekcji Qdrant `projects_docs` i wykonuje
  wyszukiwanie semantyczne. Serwis działa jako proces w tle i utrzymuje model
  w pamięci RAM/VRAM. Plik wskazany przez --index-json przechowuje stan lokalny:
  hashe plików, liczbę fragmentów i źródła.

WYMAGANIA
  - Qdrant musi działać pod http://127.0.0.1:6333.
  - Serwis uruchamia się automatycznie w tle na żądanie przy pierwszym poleceniu.
  - --index-json jest wymagany dla każdego polecenia poza --help.

SKŁADNIA
  rag_qdrant PATH --source NAME --index-json FILE [OPTIONS]
  rag_qdrant PATH --source NAZWA --index-json PLIK
  rag_qdrant --status --index-json PLIK [--json]
  rag_qdrant --list-sources --index-json PLIK [--json]
  rag_qdrant search ZAPYTANIE --index-json PLIK [--source TAGI] [--limit N] --json


ZARZĄDZANIE SERWISEM
  rag_qdrant service start    Uruchamia serwis w tle
  rag_qdrant service stop     Zatrzymuje działający serwis
  rag_qdrant service status   Sprawdza stan serwisu w tle

INDEKSOWANIE
  PATH                    Wymagany katalog główny dokumentów.
  -s, --source NAZWA      Wymagany tag źródła, np. project-a. Normalizowany do małych liter.
  --index-json PLIK       Wymagany plik JSON stanu indeksu.

WYSZUKIWANIE
  search ZAPYTANIE        Wymagane zapytanie semantyczne.
  -s, --source TAGI       Opcjonalny tag lub tagi rozdzielone przecinkami.
  --limit N               Maksymalna liczba wyników; domyślnie 5, minimum 1.
  --json                  Wymagany. Zwraca tablicę wyników JSON.
"""

PLAIN_HELP_TEXT = HELP_TEXT


def format_bytes(bytes_val: int) -> str:
    if bytes_val < 1024:
        return f"{bytes_val} B"
    elif bytes_val < 1024 * 1024:
        return f"{bytes_val / 1024:.1f} KB"
    else:
        return f"{bytes_val / (1024 * 1024):.2f} MB"


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def print_json(value: Any) -> None:
    """Emit stable machine-readable JSON response."""
    print(json.dumps(value, ensure_ascii=False))


def print_help():
    try:
        from rich.console import Console
        Console().print(HELP_TEXT)
    except Exception:
        print(PLAIN_HELP_TEXT)


def show_sources_table(sources: List[Dict[str, Any]]):
    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()
        if not sources:
            console.print("[yellow]No sources indexed yet. Run 'rag_qdrant <PATH>' to index a folder.[/yellow]")
            return

        table = Table(title="[bold green]Indexed Knowledge Sources in Qdrant[/bold green]")
        table.add_column("Source Tag", style="cyan", no_wrap=True)
        table.add_column("Indexed Files", justify="right", style="magenta")
        table.add_column("Chunks / Vectors", justify="right", style="green")
        table.add_column("Last Updated", style="dim")

        for s in sources:
            table.add_row(
                s["source"],
                str(s["files_count"]),
                str(s["chunks_count"]),
                s["last_updated"],
            )
        console.print(table)
    except Exception:
        if not sources:
            print("No sources indexed yet.")
            return
        print("-" * 60)
        print(f"{'Source Tag':<15} {'Files':<10} {'Vectors':<10} {'Last Updated'}")
        print("-" * 60)
        for s in sources:
            print(f"{s['source']:<15} {s['files_count']:<10} {s['chunks_count']:<10} {s['last_updated']}")
        print("-" * 60)


def show_status(status_payload: Dict[str, Any], sources: List[Dict[str, Any]]):
    try:
        from rich.console import Console
        console = Console()
        console.print(f"[bold green]Qdrant Status:[/bold green] Connected to {status_payload.get('qdrant_url', QDRANT_URL)}")
        console.print(f"Collection: [cyan]{status_payload.get('collection', COLLECTION_NAME)}[/cyan]")
        console.print(f"Total Vectors: [bold green]{status_payload.get('total_vectors', 0)}[/bold green]")
        console.print(f"Collection Status: {status_payload.get('health_status', 'unknown')}")
        console.print()
        show_sources_table(sources)
    except Exception:
        print(f"Qdrant Status: Connected to {status_payload.get('qdrant_url', QDRANT_URL)}")
        print(f"Collection: {status_payload.get('collection', COLLECTION_NAME)}")
        print(f"Total Vectors: {status_payload.get('total_vectors', 0)}")
        print(f"Collection Status: {status_payload.get('health_status', 'unknown')}")


def get_client(json_mode: bool = False) -> RagServiceClient:
    """Create client and ensure service is active."""
    client = RagServiceClient()
    if not client.is_ready():
        def notify(msg: str):
            if json_mode:
                sys.stderr.write(f"[rag_qdrant] {msg}\n")
                sys.stderr.flush()
            else:
                try:
                    from rich.console import Console
                    Console().print(f"[dim]{msg}[/dim]")
                except Exception:
                    print(msg)
        try:
            client.ensure_service_running(notify_cb=notify)
        except Exception as e:
            if json_mode:
                print_json({"error": f"Failed to connect to rag_qdrant background service: {e}"})
                sys.exit(1)
            else:
                print(f"[Error] Failed to connect to rag_qdrant service: {e}")
                sys.exit(1)
    return client


def handle_service_command(args: List[str]) -> int:
    """Manage background service explicitly."""
    subcmd = args[0] if args else "status"
    client = RagServiceClient()

    if subcmd == "start":
        if client.is_ready():
            print(f"rag_qdrant service is already running on {SERVICE_URL}.")
            return 0
        print("Starting rag_qdrant service in background...")
        try:
            client.ensure_service_running(notify_cb=print)
            print(f"Service started successfully on {SERVICE_URL}.")
            return 0
        except Exception as e:
            print(f"[Error] Failed to start service: {e}")
            return 1

    elif subcmd == "stop":
        if not client.is_healthy():
            print("rag_qdrant service is not running.")
            return 0
        try:
            client.stop_service()
            print("rag_qdrant service stopped.")
            return 0
        except Exception as e:
            print(f"[Error] Failed to stop service: {e}")
            return 1

    elif subcmd == "status":
        if client.is_ready():
            print(f"rag_qdrant service is RUNNING and READY on {SERVICE_URL}.")
            try:
                status = client.get_status()
                print(f"  • Collection: {status.get('collection')}")
                print(f"  • Total Vectors: {status.get('total_vectors')}")
                print(f"  • Total Files: {status.get('total_files')}")
            except Exception:
                pass
            return 0
        elif client.is_healthy():
            print(f"rag_qdrant service is STARTING UP on {SERVICE_URL} (initializing model)...")
            return 0
        else:
            print(f"rag_qdrant service is NOT running on {SERVICE_URL}.")
            return 0

    else:
        print(f"Unknown service command: '{subcmd}'. Use 'start', 'stop', or 'status'.")
        return 1


def main_search(argv: List[str]) -> int:
    """Execute semantic search as a machine-readable CLI command."""
    args = create_search_parser().parse_args(argv)
    if not args.json:
        print_json({"error": "The search command requires --json."})
        return 1
    if not args.index_json:
        print_json({"error": "The --index-json option is required and must name the JSON index file."})
        return 1
    if args.limit < 1:
        print_json({"error": "--limit must be at least 1."})
        return 1

    try:
        client = get_client(json_mode=True)
        results = client.search(
            query=args.query,
            sources=args.source,
            limit=args.limit,
            index_json=args.index_json,
        )
    except ServiceError as error:
        print_json({"error": str(error)})
        return 1
    except Exception as error:
        print_json({"error": str(error)})
        return 1

    print_json(results)
    return 0


def main() -> int:
    if len(sys.argv) <= 1:
        print_help()
        return 0

    if sys.argv[1] == "search":
        return main_search(sys.argv[2:])

    if sys.argv[1] == "service":
        return handle_service_command(sys.argv[2:])

    parser = create_parser()

    try:
        args, unknown = parser.parse_known_args()
    except Exception:
        print_help()
        return 1

    if args.help:
        print_help()
        return 0

    if args.json and not (args.list_sources or args.status):
        print_json({"error": "--json is supported only with --status, --list-sources, or search."})
        return 1

    if not args.index_json:
        error = "The --index-json option is required and must name the JSON index file."
        if args.json:
            print_json({"error": error})
        else:
            print(f"[Error] {error}")
            print_help()
        return 1

    index_json_str = args.index_json

    target_path_str = args.path
    if target_path_str:
        target_path_str = target_path_str.strip('"\'; ')
    elif unknown:
        for u in unknown:
            cleaned = u.strip('"\'; ')
            if Path(cleaned).is_dir():
                target_path_str = cleaned
                unknown.remove(u)
                break

    if unknown and not (args.list_sources or args.status):
        print(f"\n[Error] Unknown option(s): {' '.join(unknown)}\n")
        print_help()
        return 1

    client = get_client(json_mode=args.json)

    if args.list_sources:
        try:
            sources = client.get_sources(index_json=index_json_str)
        except Exception as error:
            if args.json:
                print_json({"error": str(error)})
            else:
                print(f"[Error] Failed to list sources: {error}")
            return 1

        if args.json:
            print_json(sources)
        else:
            show_sources_table(sources)
        return 0

    if args.status:
        try:
            status_payload = client.get_status(index_json=index_json_str)
            sources = client.get_sources(index_json=index_json_str)
        except Exception as error:
            if args.json:
                print_json({"error": str(error)})
            else:
                print(f"[Status Error] Failed to connect to Qdrant: {error}")
            return 1

        if args.json:
            print_json(status_payload)
        else:
            show_status(status_payload, sources)
        return 0

    if not target_path_str:
        print_help()
        return 0

    target_dir = Path(target_path_str).resolve()
    if not target_dir.is_dir():
        print(f"[Error] Target path '{target_path_str}' does not exist or is not a directory.")
        return 1

    indexing_error = validate_indexing_arguments(args)
    if indexing_error:
        print(
            f"\n[Error] {indexing_error} "
            "Example: rag_qdrant <PATH> --source project-a --index-json D:\\rag-index.json\n"
        )
        return 1

    source_name = args.source.strip().lower()

    try:
        from rich.console import Console
        from rich.progress import Progress, BarColumn, TextColumn
        from rich.markup import escape
        console = Console()
        console.print(f"[bold cyan]─── RAG Indexing Configuration ──────────────────────────[/bold cyan]")
        console.print(f"  • Target Directory:       [bold]{target_dir}[/bold]")
        console.print(f"  • Source Tag:             [bold green]{source_name}[/bold green]")
        console.print(f"  • Qdrant URL:             {QDRANT_URL}")
        console.print(f"  • Qdrant Collection:      [bold]{COLLECTION_NAME}[/bold]")
        console.print(f"  • Index JSON File:        [bold magenta]{index_json_str}[/bold magenta]")
        console.print(f"  • Service URL:            [bold]{SERVICE_URL}[/bold]")
        console.print(f"[bold cyan]──────────────────────────────────────────────────────────[/bold cyan]\n")

        start_time = time.time()
        console.print("[bold yellow]Connecting to RAG service and initiating indexing...[/bold yellow]")
        sys.stdout.flush()

        final_stats = None
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TextColumn("({task.completed}/{task.total})"),
            console=console,
        ) as progress:
            task_id = progress.add_task("Preparing...", total=100)
            last_reported = {"action": "", "pct": -1.0}

            for event in client.index_directory_stream(
                directory=target_dir,
                source=source_name,
                index_json=index_json_str,
            ):
                ev_type = event.get("type")

                if ev_type == "plan":
                    plan = event.get("plan", {})
                    console.print("\n[bold cyan]─── Indexing Execution Plan ───────────────────────────────[/bold cyan]")
                    console.print(f"  • Total Scanned:         {plan.get('scanned_files', 0)} files ({format_bytes(plan.get('scanned_bytes', 0))})")
                    console.print(f"  • Files to Index/Update: [bold green]{plan.get('to_index_files', 0)}[/bold green] files ([bold green]{format_bytes(plan.get('to_index_bytes', 0))}[/bold green])")
                    console.print(f"  • Files Unchanged:       {plan.get('skipped_files', 0)} files ({format_bytes(plan.get('skipped_bytes', 0))})")
                    console.print(f"  • Compute Engine:        [bold green]RAG Service[/bold green] ({plan.get('cpu_workers', NUM_WORKERS)} CPU threads for chunking)")
                    console.print("[bold cyan]────────────────────────────────────────────────────────────[/bold cyan]\n")
                    sys.stdout.flush()

                elif ev_type == "progress":
                    completed = event.get("completed", 0)
                    total = event.get("total", 1)
                    filename = event.get("filename", "")
                    action = event.get("action", "")
                    is_bytes = event.get("is_bytes", True)
                    count_str = event.get("count_str", "")

                    pct = (completed / total * 100.0) if total > 0 else 100.0
                    if is_bytes:
                        desc = f"[{action}] {format_bytes(completed)}/{format_bytes(total)} - {filename[:25]}"
                    else:
                        desc = f"[{action}] {completed}/{total} - {filename[:25]}"
                    progress.update(task_id, total=total, completed=completed, description=desc)

                    if last_reported["action"] != action:
                        last_reported["action"] = action
                        last_reported["pct"] = -1.0

                    step_threshold = 2.0 if action == "indexing" else 5.0
                    if last_reported["pct"] < 0 or (pct - last_reported["pct"] >= step_threshold) or completed == total:
                        last_reported["pct"] = pct
                        elapsed = time.time() - start_time
                        time_str = f"[dim]{format_time(elapsed)}[/dim]"
                        col_w = max(len(count_str), 9) if count_str else 9
                        sp = " " * col_w
                        count_col = f"| {count_str:>{col_w}} | " if count_str else f"| {sp} | "
                        if is_bytes:
                            prefix = f"  {time_str} [cyan][{action.capitalize():<10} {pct:5.1f}%][/cyan] {format_bytes(completed):>9} / {format_bytes(total):<9} {count_col}"
                        else:
                            prefix = f"  {time_str} [cyan][{action.capitalize():<10} {pct:5.1f}%][/cyan] {completed:>9} / {total:<9} {count_col}"
                        console.print(prefix + escape(filename), highlight=False)
                        sys.stdout.flush()

                elif ev_type == "complete":
                    final_stats = event.get("stats", {})

                elif ev_type == "error":
                    raise ServiceError(event.get("error", "Unknown error during indexing."))

        total_elapsed = time.time() - start_time
        console.print("\n[bold green]Indexing Complete![/bold green]")
        console.print(f"  • Total Time Elapsed:    {format_time(total_elapsed)}")
        if final_stats:
            console.print(f"  • Files Scanned:         {final_stats.get('scanned', 0)}")
            console.print(f"  • Newly Indexed:         {final_stats.get('indexed', 0)}")
            console.print(f"  • Updated:               {final_stats.get('updated', 0)}")
            console.print(f"  • Skipped (unchanged):   {final_stats.get('skipped', 0)}")
            console.print(f"  • Deleted from Qdrant:   {final_stats.get('deleted', 0)}")
            console.print(f"  • Total Vectors Added:   {final_stats.get('total_points', 0)}")
        console.print()
        sys.stdout.flush()

        try:
            status_payload = client.get_status(index_json=index_json_str)
            sources = client.get_sources(index_json=index_json_str)
            show_status(status_payload, sources)
        except Exception:
            pass

        return 0

    except KeyboardInterrupt:
        print("\n\n[Cancelled] Indexing interrupted by user.\n")
        return 130
    except Exception as e:
        print(f"\n[Error] Indexing failed: {e}\n")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n[Cancelled] Operation interrupted by user.\n")
        os._exit(130)
