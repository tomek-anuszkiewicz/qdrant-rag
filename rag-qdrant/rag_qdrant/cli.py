import io
import json
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

# Ensure unbuffered UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

import logging
import warnings
logging.getLogger("google_genai").setLevel(logging.ERROR)
try:
    from google.genai.models import Models, AsyncModels
    Models._logged_afc_warning = True
    AsyncModels._logged_afc_warning = True
except Exception:
    pass

warnings.filterwarnings("ignore", message=".*automatic function calling.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*genai.*")

try:
    from .config import QDRANT_URL, COLLECTION_NAME, CACHE_FILE, NUM_WORKERS, VISION_MAX_WORKERS
    from .arguments import create_parser, create_search_parser, validate_indexing_arguments
    from .indexer import KnowledgeIndexer
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from rag_qdrant.config import QDRANT_URL, COLLECTION_NAME, CACHE_FILE, NUM_WORKERS, VISION_MAX_WORKERS
    from rag_qdrant.arguments import create_parser, create_search_parser, validate_indexing_arguments
    from rag_qdrant.indexer import KnowledgeIndexer


HELP_TEXT = """
[bold cyan]rag_qdrant[/bold cyan] - Multi-Source Local RAG Indexer (Qdrant)

[bold yellow]USAGE:[/bold yellow]
  rag_qdrant PATH --source NAME --include-dirs DIRS [OPTIONS]
  rag_qdrant --status | --list-sources
  rag_qdrant search QUERY [--source NAME] [--limit COUNT] --json

[bold yellow]ARGUMENTS:[/bold yellow]
  [green]PATH[/green]                   Directory to index (e.g. [bold].[/bold] for current folder, or [bold]<PATH>[/bold])

[bold yellow]OPTIONS:[/bold yellow]
  [green]-s, --source NAME[/green]      [bold red][REQUIRED][/bold red] Tag for the indexed source (e.g. 'project-a', 'engineering-notes').
  [green]--include-dirs DIRS[/green]    [bold red][REQUIRED FOR INDEXING][/bold red] Top-level directories to index (e.g. '01*' '02*').
  [green]-l, --list-sources[/green]     Display a table of all indexed sources with file & vector counts.
  [green]--status[/green]               Check Qdrant database connectivity and total collection size.
  [green]--json[/green]                 Emit machine-readable JSON for status or source listing.
  [green]--reindex[/green]              Force re-indexing of all files (ignores SHA256 cache).
  [green]-h, --help[/green]             Show this help message and exit.

[bold yellow]EXAMPLES:[/bold yellow]
  rag_qdrant <PATH_TO_DOCUMENTS> --source project-a --include-dirs "docs" "design"
  rag_qdrant <PATH_TO_NOTES> --source engineering-notes --include-dirs "01*" "02*"
  rag_qdrant --list-sources
  rag_qdrant --status
  rag_qdrant search "DMA arbitration" --source project-a --json
"""

PLAIN_HELP_TEXT = """
rag_qdrant - Multi-Source Local RAG Indexer (Qdrant)

USAGE:
  rag_qdrant PATH --source NAME --include-dirs DIRS [OPTIONS]
  rag_qdrant --status | --list-sources
  rag_qdrant search QUERY [--source NAME] [--limit COUNT] --json

ARGUMENTS:
  PATH                   Directory to index (e.g. '.' for current folder, or <PATH>)

OPTIONS:
  -s, --source NAME      [REQUIRED] Tag for the indexed source (e.g. 'project-a', 'engineering-notes').
  --include-dirs DIRS    [REQUIRED FOR INDEXING] Top-level directories to index (e.g. '01*' '02*').
  -l, --list-sources     Display a table of all indexed sources with file & vector counts.
  --status               Check Qdrant database connectivity and total collection size.
  --json                 Emit machine-readable JSON for status or source listing.
  --reindex              Force re-indexing of all files (ignores SHA256 cache).
  -h, --help             Show this help message and exit.

EXAMPLES:
  rag_qdrant <PATH_TO_DOCUMENTS> --source project-a --include-dirs "docs" "design"
  rag_qdrant <PATH_TO_NOTES> --source engineering-notes --include-dirs "01*" "02*"
  rag_qdrant --list-sources
  rag_qdrant --status
  rag_qdrant search "DMA arbitration" --source project-a --json
"""


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


def print_collection_stats(indexer: KnowledgeIndexer, title: str = "Qdrant Collection Status"):
    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()
        stats = indexer.get_qdrant_sources_stats()

        table = Table(title=f"[bold cyan]{title}[/bold cyan] (Collection: [green]{COLLECTION_NAME}[/green])")
        table.add_column("Source Tag", style="cyan", no_wrap=True)
        table.add_column("Qdrant Points", justify="right", style="green")
        table.add_column("Cached Files", justify="right", style="magenta")

        for s, data in stats["sources"].items():
            table.add_row(s, f"{data['vectors']:,}", str(data['cached_files']))

        table.add_section()
        table.add_row("[bold]Total[/bold]", f"[bold green]{stats['total_points']:,}[/bold green]", f"[bold magenta]{stats['total_files']}[/bold magenta]")
        console.print(table)
        console.print()
    except Exception as e:
        print(f"[{title}] Failed to retrieve Qdrant stats: {e}")


def print_help():
    try:
        from rich.console import Console
        console = Console()
        console.print(HELP_TEXT)
    except Exception:
        print(PLAIN_HELP_TEXT)


def show_sources_table(indexer: KnowledgeIndexer):
    sources = indexer.get_sources_stats()
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
                s["last_updated"]
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


def show_status(indexer: KnowledgeIndexer):
    try:
        from rich.console import Console
        console = Console()
        indexer.ensure_collection()
        info = indexer.client.get_collection(COLLECTION_NAME)
        console.print(f"[bold green]Qdrant Status:[/bold green] Connected to {QDRANT_URL}")
        console.print(f"Collection: [cyan]{COLLECTION_NAME}[/cyan]")
        console.print(f"Total Vectors: [bold green]{info.points_count}[/bold green]")
        console.print(f"Collection Status: {info.status}")
    except Exception as e:
        print(f"[Status Error] Failed to connect to Qdrant at {QDRANT_URL}: {e}")


def create_indexer(json_output: bool = False) -> KnowledgeIndexer:
    """Initialize the indexer without contaminating a JSON command response."""
    if not json_output:
        return KnowledgeIndexer()

    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return KnowledgeIndexer()


def get_status_payload(indexer: KnowledgeIndexer) -> dict:
    """Return the status data consumed by PATH-based MCP servers."""
    indexer.ensure_collection()
    info = indexer.client.get_collection(COLLECTION_NAME)
    sources = indexer.get_sources_stats()
    return {
        "qdrant_url": QDRANT_URL,
        "collection": COLLECTION_NAME,
        "health_status": str(info.status),
        "total_vectors": info.points_count,
        "total_files": sum(source["files_count"] for source in sources),
        "source_count": len(sources),
    }


def print_json(value) -> None:
    """Emit the stable machine-readable CLI response."""
    print(json.dumps(value, ensure_ascii=False))


def main_search(argv) -> int:
    """Execute semantic search as a machine-readable CLI command."""
    args = create_search_parser().parse_args(argv)
    if not args.json:
        print_json({"error": "The search command requires --json."})
        return 1
    if args.limit < 1:
        print_json({"error": "--limit must be at least 1."})
        return 1

    try:
        indexer = create_indexer(json_output=True)
        results = indexer.search(query=args.query, sources=args.source, limit=args.limit)
    except Exception as error:
        print_json({"error": str(error)})
        return 1

    print_json(results)
    return 0


def main():
    if len(sys.argv) <= 1:
        print_help()
        sys.exit(0)

    if sys.argv[1] == "search":
        return main_search(sys.argv[2:])

    parser = create_parser()

    try:
        args, unknown = parser.parse_known_args()
    except Exception:
        print_help()
        sys.exit(1)

    if args.help:
        print_help()
        sys.exit(0)

    if args.json and not (args.list_sources or args.status):
        print_json({"error": "--json is supported only with --status, --list-sources, or search."})
        return 1

    # Clean path string from accidental trailing quotes or slashes
    target_path_str = args.path
    if target_path_str:
        target_path_str = target_path_str.strip('"\'; ')
    elif unknown:
        # Check if an unknown arg is actually an existing directory
        for u in unknown:
            cleaned = u.strip('"\'; ')
            if Path(cleaned).is_dir():
                target_path_str = cleaned
                unknown.remove(u)
                break

    if unknown and not (args.list_sources or args.status):
        print(f"\n[Error] Unknown option(s): {' '.join(unknown)}\n")
        print_help()
        sys.exit(1)

    try:
        indexer = create_indexer(json_output=args.json)
    except Exception as e:
        if args.json:
            print_json({"error": str(e)})
            return 1
        print(f"[Error] Initialization failed: {e}")
        sys.exit(1)

    if args.list_sources:
        if args.json:
            print_json(indexer.get_sources_stats())
        else:
            show_sources_table(indexer)
        return 0

    if args.status:
        if args.json:
            try:
                print_json(get_status_payload(indexer))
            except Exception as error:
                print_json({"error": str(error)})
                return 1
        else:
            show_status(indexer)
            show_sources_table(indexer)
        return 0

    if not target_path_str:
        print_help()
        return

    target_dir = Path(target_path_str).resolve()
    if not target_dir.is_dir():
        print(f"[Error] Target path '{target_path_str}' does not exist or is not a directory.")
        sys.exit(1)

    indexing_error = validate_indexing_arguments(args)
    if indexing_error:
        print(
            f"\n[Error] {indexing_error} "
            "Example: rag_qdrant <PATH> --source project-a --include-dirs \"docs\"\n"
        )
        sys.exit(1)
    source_name = args.source.strip().lower()


    try:
        from rich.console import Console
        from rich.progress import Progress, BarColumn, TextColumn
        from rich.markup import escape
        console = Console()
        console.print(f"[bold cyan]─── RAG Indexing Configuration ──────────────────────────[/bold cyan]")
        console.print(f"  • Target Directory:       [bold]{target_dir}[/bold]")
        console.print(f"  • Source Tag:             [bold green]{source_name}[/bold green]")
        if args.include_dirs:
            console.print(f"  • Included Subdirs:       [bold green]{', '.join(args.include_dirs)}[/bold green]")
        console.print(f"  • Qdrant URL:             {QDRANT_URL}")
        console.print(f"  • Qdrant Collection:      [bold]{COLLECTION_NAME}[/bold]")
        console.print(f"  • Hash Cache File:        [bold magenta]{CACHE_FILE}[/bold magenta]")
        provider_style = "bold green" if indexer.active_provider == "CUDA" else "bold yellow"
        console.print(f"  • Embedder Engine:        [{provider_style}]{indexer.active_provider}[/{provider_style}] (Batch Size: {indexer.active_batch_size})")
        console.print(f"  • Hashing & Chunks:       [bold yellow]{NUM_WORKERS} CPU threads[/bold yellow]")
        console.print(f"  • Diagram Vision:         [bold green]Offline Sidecar Loader (<image>.txt)[/bold green]")
        console.print(f"[bold cyan]──────────────────────────────────────────────────────────[/bold cyan]\n")

        if indexer.active_provider != "CUDA":
            from rich.panel import Panel
            advisory_lines = []
            host_gpu = getattr(indexer, "host_gpu", None)
            if host_gpu:
                advisory_lines.append(f"  [bold green]Discrete GPU Detected:[/bold green] {host_gpu}")
                advisory_lines.append(f"  [yellow]Status:[/yellow] Running on CPU because ONNX CUDA runtime libraries (cublasLt64) were not loaded.")
                advisory_lines.append(f"  [bold cyan]To unlock 5-10x faster RTX acceleration (installs missing cublasLt64 DLL):[/bold cyan]")
                advisory_lines.append(f"    [white]pip install nvidia-cublas-cu12[/white]")
            else:
                advisory_lines.append(f"  [bold yellow]No discrete NVIDIA GPU detected.[/bold yellow] Running on multi-core CPU ({NUM_WORKERS} threads).")

            advisory_lines.append("")
            advisory_lines.append(f"  [bold magenta]Cloud API Alternative:[/bold magenta]")
            advisory_lines.append(f"    Have high API token quotas? Cloud embeddings can be enabled via GEMINI_API_KEY in .env.")

            console.print(Panel("\n".join(advisory_lines), title="[bold yellow]💡 Compute Acceleration Advisory[/bold yellow]", border_style="yellow"))
            console.print()

        if args.reindex:
            console.print("[yellow]Forced re-indexing enabled (cache ignored).[/yellow]\n")

        print_collection_stats(indexer, "Initial Qdrant Collection State")

        start_time = time.time()
        console.print("[bold yellow]Scanning & computing SHA256 hashes...[/bold yellow]")
        sys.stdout.flush()

        def plan_callback(plan):
            console.print("\n[bold cyan]─── Indexing Execution Plan ───────────────────────────────[/bold cyan]")
            console.print(f"  • Total Scanned:         {plan['scanned_files']} files ({format_bytes(plan['scanned_bytes'])})")
            console.print(f"  • Files to Index/Update: [bold green]{plan['to_index_files']}[/bold green] files ([bold green]{format_bytes(plan['to_index_bytes'])}[/bold green])")
            console.print(f"  • Files Unchanged:       {plan['skipped_files']} files ({format_bytes(plan['skipped_bytes'])})")
            engine_style = "bold green" if indexer.active_provider == "CUDA" else "bold yellow"
            console.print(f"  • Compute Engine:        [{engine_style}]{indexer.active_provider}[/{engine_style}] (Batch Size: {indexer.active_batch_size}), [bold yellow]{plan['cpu_workers']} CPU threads[/bold yellow] for chunking")
            console.print("[bold cyan]────────────────────────────────────────────────────────────[/bold cyan]\n")
            sys.stdout.flush()


        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TextColumn("({task.completed}/{task.total})"),
            console=console
        ) as progress:
            task_id = progress.add_task("Preparing...", total=100)
            last_reported = {"action": "", "pct": -1.0}

            def progress_callback(completed, total, filename, action, is_bytes=True, count_str=""):
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

            stats = indexer.index_directory(
                directory=target_dir,
                source_name=source_name,
                force=args.reindex,
                include_dirs=args.include_dirs,
                progress_cb=progress_callback,
                plan_cb=plan_callback
            )

        total_elapsed = time.time() - start_time
        console.print("\n[bold green]Indexing Complete![/bold green]")
        console.print(f"  • Total Time Elapsed:    {format_time(total_elapsed)}")
        console.print(f"  • Files Scanned:         {stats['scanned']}")
        console.print(f"  • Newly Indexed:         {stats['indexed']}")
        console.print(f"  • Updated:               {stats['updated']}")
        console.print(f"  • Skipped (unchanged):   {stats['skipped']}")
        console.print(f"  • Deleted from Qdrant:   {stats['deleted']}")
        console.print(f"  • Total Vectors Added:   {stats['total_points']}")
        if stats["images_analyzed"] > 0:
            console.print(f"  • Diagrams/OCR Analyzed: {stats['images_analyzed']}")
        console.print()
        sys.stdout.flush()

        print_collection_stats(indexer, "Final Qdrant Collection State")
    except KeyboardInterrupt:
        elapsed = time.time() - start_time if 'start_time' in locals() else 0.0
        console.print(f"\n\n[bold yellow]⚠ Indexing cancelled by user after {format_time(elapsed)} (Ctrl+C).[/bold yellow]")
        console.print("[dim]All files completed up to this point were saved to Qdrant and cached.[/dim]\n")
        try:
            print_collection_stats(indexer, "Current Qdrant Collection State")
        except Exception:
            pass
        sys.stdout.flush()
        import os
        os._exit(130)
    except Exception as e:
        console.print(f"\n[bold red][Error] Indexing failed:[/bold red] {e}")
        sys.exit(1)



if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        try:
            from rich.console import Console
            Console().print("\n\n[bold yellow]⚠ Operation cancelled by user (Ctrl+C).[/bold yellow]\n")
        except Exception:
            print("\n\n[Cancelled] Operation interrupted by user.\n")
        import os
        os._exit(130)

