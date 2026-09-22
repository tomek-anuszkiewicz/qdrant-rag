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

try:
    from .config import QDRANT_URL, COLLECTION_NAME, NUM_WORKERS
    from .arguments import create_parser, create_search_parser, validate_indexing_arguments
    from .indexer import KnowledgeIndexer
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from rag_qdrant.config import QDRANT_URL, COLLECTION_NAME, NUM_WORKERS
    from rag_qdrant.arguments import create_parser, create_search_parser, validate_indexing_arguments
    from rag_qdrant.indexer import KnowledgeIndexer


HELP_TEXT = """
rag_qdrant — lokalny, przyrostowy indeksator Markdown dla Qdrant

CEL
  Indeksuje dokumenty Markdown do kolekcji Qdrant `projects_docs` i wykonuje
  wyszukiwanie semantyczne. Wektory są zapisywane w Qdrant pod
  http://localhost:6333. Plik wskazany przez --index-json przechowuje wyłącznie
  lokalny stan: hashe plików, liczbę fragmentów i źródła.

WYMAGANIA
  - Qdrant musi działać pod http://localhost:6333.
  - Zainstaluj zależności z requirements.txt.
  - --index-json jest wymagany dla każdego polecenia poza --help. Ten sam plik
    JSON należy przekazywać we wszystkich przebiegach obsługujących tę kolekcję.
    Plik może jeszcze nie istnieć; zostanie utworzony przy pierwszym zapisie.

SKŁADNIA
  rag_qdrant PATH --source NAZWA --index-json PLIK
  rag_qdrant --status --index-json PLIK [--json]
  rag_qdrant --list-sources --index-json PLIK [--json]
  rag_qdrant search ZAPYTANIE --index-json PLIK [--source TAGI] [--limit N] --json

INDEKSOWANIE
  PATH                    Wymagany katalog główny dokumentów.
  -s, --source NAZWA      Wymagany tag źródła, np. project-a. Jest normalizowany
                          do małych liter i służy do filtrowania wyszukiwania.
  --index-json PLIK       Wymagany plik JSON stanu indeksu.

  Skanowane są wszystkie pliki .md w PATH — także w katalogu głównym oraz we
  wszystkich podkatalogach. Pomijane są tylko katalogi techniczne/prywatne,
  m.in. .git, .obsidian, .venv, node_modules, __pycache__ i nazwy z `private`.

  Każdy znaleziony plik jest ponownie haszowany SHA-256 przy każdym przebiegu:
  - nowy plik: jest dzielony na fragmenty, wektory są dodawane do Qdrant;
  - zmieniony hash: stare punkty pliku są usuwane, potem zapisywane są nowe;
  - identyczny hash: plik jest pomijany — nie tworzy fragmentów ani embeddingów;
  - plik usunięty z bieżącego PATH: jego punkty i wpis JSON są usuwane.
  Nie ma trybu pełnego wymuszonego reindeksowania.

OBRAZY
  Obrazy nie są analizowane, opisywane, odczytywane z sidecarów ani przekazywane
  do usług chmurowych. Ich ścieżki mogą pozostać metadanymi fragmentu Markdown,
  ale tekst obrazu nie wpływa na embedding ani wynik wyszukiwania.

ODCZYT STANU
  --status                Sprawdza Qdrant i pokazuje stan kolekcji. Bez --json
                          pokazuje również tabelę źródeł.
  -l, --list-sources      Pokazuje dane źródeł z --index-json: tag, liczbę
                          plików, fragmentów i czas ostatniego indeksowania.
  --json                  Dla --status i --list-sources zwraca odpowiedź JSON.
                          W głównym trybie nie używaj go z indeksowaniem.

WYSZUKIWANIE
  search ZAPYTANIE        Wymagane zapytanie semantyczne.
  -s, --source TAGI       Opcjonalny tag lub tagi rozdzielone przecinkami.
                          Bez niego przeszukiwana jest cała kolekcja.
  --limit N               Maksymalna liczba wyników; domyślnie 5, minimum 1.
  --json                  Wymagany. Zwraca tablicę wyników z score, source,
                          file_path, relative_path, header, content i images.
                          Wyniki o score niższym niż 0.50 nie są zwracane.

PRZYKŁADY
  rag_qdrant D:\\Docs\\Projekt --source project-a --index-json D:\\AI\\qdrant\\rag-index.json
  rag_qdrant --status --index-json D:\\AI\\qdrant\\rag-index.json --json
  rag_qdrant --list-sources --index-json D:\\AI\\qdrant\\rag-index.json --json
  rag_qdrant search "DMA arbitration" --source project-a,notes --limit 10 --index-json D:\\AI\\qdrant\\rag-index.json --json

AUTOMATYZACJA
  Agent powinien używać --json dla statusu, listy źródeł i wyszukiwania oraz
  parsować stdout jako JSON. Indeksowanie jest interaktywne i wypisuje postęp
  tekstowy; po powodzeniu jego zmiany są trwałe w Qdrant i --index-json.
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


def create_indexer(index_json: Path, json_output: bool = False) -> KnowledgeIndexer:
    """Initialize the indexer without contaminating a JSON command response."""
    if not json_output:
        return KnowledgeIndexer(index_json)

    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return KnowledgeIndexer(index_json)


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
    if not args.index_json:
        print_json({"error": "The --index-json option is required and must name the JSON index file."})
        return 1
    if args.limit < 1:
        print_json({"error": "--limit must be at least 1."})
        return 1

    try:
        indexer = create_indexer(Path(args.index_json).expanduser().resolve(), json_output=True)
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

    if not args.index_json:
        error = "The --index-json option is required and must name the JSON index file."
        if args.json:
            print_json({"error": error})
        else:
            print(f"[Error] {error}")
            print_help()
        return 1
    index_json = Path(args.index_json).expanduser().resolve()

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
        indexer = create_indexer(index_json, json_output=args.json)
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
            "Example: rag_qdrant <PATH> --source project-a --index-json D:\\rag-index.json\n"
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
        console.print(f"  • Qdrant URL:             {QDRANT_URL}")
        console.print(f"  • Qdrant Collection:      [bold]{COLLECTION_NAME}[/bold]")
        console.print(f"  • Index JSON File:        [bold magenta]{index_json}[/bold magenta]")
        provider_style = "bold green" if indexer.active_provider == "CUDA" else "bold yellow"
        console.print(f"  • Embedder Engine:        [{provider_style}]{indexer.active_provider}[/{provider_style}] (Batch Size: {indexer.active_batch_size})")
        console.print(f"  • Hashing & Chunks:       [bold yellow]{NUM_WORKERS} CPU threads[/bold yellow]")
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

            console.print(Panel("\n".join(advisory_lines), title="[bold yellow]💡 Compute Acceleration Advisory[/bold yellow]", border_style="yellow"))
            console.print()

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

