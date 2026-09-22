import re
import sys
import argparse
from pathlib import Path

# Ensure the rag-qdrant tool root is in the Python path.
tools_rag_dir = Path(__file__).resolve().parent
if str(tools_rag_dir) not in sys.path:
    sys.path.insert(0, str(tools_rag_dir))

from rag_qdrant.assets_manager import AssetsManager

def clean_markdown_text(text: str) -> str:
    """Strips excessive markdown symbols, tables and wiki links for concise text."""
    t = re.sub(r'\[\[(.*?)\]\]', r'\1', text)
    t = re.sub(r'#+\s*', '', t)
    t = re.sub(r'>\s*\[!.*?\]\s*.*', '', t)
    t = re.sub(r'`+', '', t)
    t = re.sub(r'\|', ' ', t)
    t = re.sub(r'\s+', ' ', t)
    return t.strip()


def extract_context_for_image(img_name: str, book_dir: Path):
    """Finds referencing markdown file and extracts caption and context surrounding the image."""
    for md_file in sorted(book_dir.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8", errors="replace")
        pattern = r'(!\[([^\]]*)\]\([^)]*' + re.escape(img_name) + r'[^)]*\)|!\[\[[^\]]*' + re.escape(img_name) + r'[^\]]*\]\])'
        match = re.search(pattern, content)
        if match:
            caption = match.group(2).strip() if match.group(2) else ""
            if not caption:
                # Attempt to find caption from preceding or succeeding line
                start_idx = match.start()
                preceding = content[max(0, start_idx - 200):start_idx]
                lines = [l.strip() for l in preceding.splitlines() if l.strip()]
                if lines:
                    caption = lines[-1].strip("*_# ")

            start = max(0, match.start() - 600)
            end = min(len(content), match.end() + 1000)
            surrounding = content[start:end]
            return {
                "md_name": md_file.name,
                "caption": caption or img_name.replace("_", " ").replace(".svg", "").replace(".png", ""),
                "surrounding": surrounding
            }
    return None


def deduce_signals_and_subsystem(caption: str, surrounding: str, img_name: str):
    """Deduces subsystem, signals and key takeaways from technical context."""
    text = (caption + " " + surrounding + " " + img_name).upper()
    signals = set()
    known_signals = [
        "CLK", "E-CLOCK", "E CLK", "7MHZ", "CCK", "CDAC", "/AS", "AS", "/UDS", "/LDS", "UDS", "LDS",
        "R/W", "R/W*", "/DTACK", "DTACK", "/BERR", "BERR", "/VPA", "VPA", "/VMA", "VMA", "/RESET", "RESET",
        "/HALT", "HALT", "/BR", "BR", "/BG", "BG", "/BGACK", "BGACK", "IPL0", "IPL1", "IPL2",
        "FC0", "FC1", "FC2", "A1-A23", "D0-D15", "D0-D7", "D8-D15", "BLTCON0", "BLTCON1", "DMACON",
        "AUD0DAT", "AUD1DAT", "AUD2DAT", "AUD3DAT", "BPLCON0", "BPLCON1", "BPLCON2", "COPCON"
    ]
    for s in known_signals:
        if re.search(r'\b' + re.escape(s) + r'\b', text):
            signals.add(s)

    subsystem = "General Amiga / Motorola Architecture"
    if "AGNUS" in text or "DMA" in text or "BLITTER" in text:
        subsystem = "Agnus (Blitter, Copper, DMA Channels & Memory Bus Arbitration)"
    elif "DENISE" in text or "PLAYFIELD" in text or "SPRITE" in text:
        subsystem = "Denise (Video Display, Dual Playfields, Sprites, Colors & Priorities)"
    elif "PAULA" in text or "AUDIO" in text or "FLOPPY" in text or "UART" in text or "CHINON" in text:
        subsystem = "Paula (Audio DMA Channels, Disk Controller & Serial I/O)"
    elif "8520" in text or "CIA" in text or "KEYBOARD" in text or "TIMER" in text:
        subsystem = "CIA (MOS 8520 Complex Interface Adapter, Timers & Ports)"
    elif "68000" in text or "M68000" in text or "BUS" in text or "CYCLE" in text or "ADDRESSING" in text:
        subsystem = "Motorola 68000 CPU (Bus Cycles, Addressing Modes, Exception Handling & Pinouts)"
    elif "EXPANSION" in text or "CONNECTOR" in text or "BACKPLANE" in text or "AUTOCONFIG" in text:
        subsystem = "Expansion Bus & Autoconfig Architecture (Zorro II, Edge Connectors)"

    return subsystem, sorted(list(signals))


def main():
    parser = argparse.ArgumentParser(
        description="Generate text sidecars for unindexed images below a document root."
    )
    parser.add_argument("document_root", type=Path, help="Root directory containing Markdown and images")
    args = parser.parse_args()
    document_root = args.document_root.resolve()
    if not document_root.is_dir():
        parser.error(f"Document root does not exist or is not a directory: {document_root}")

    mgr = AssetsManager()
    unindexed = mgr.get_unindexed_images(document_root)
    print(f"Total unindexed images detected by AssetsManager: {len(unindexed)}")

    created_count = 0
    for item in unindexed:
        img_p = item["path"]
        book_dir = document_root
        book_name = document_root.name

        info = extract_context_for_image(img_p.name, book_dir)
        caption = info["caption"] if info else img_p.stem.replace("_", " ").title()
        surrounding = info["surrounding"] if info else ""
        md_name = info["md_name"] if info else "Reference Manual"

        subsystem, signals = deduce_signals_and_subsystem(caption, surrounding, img_p.name)

        clean_context = clean_markdown_text(surrounding)
        if len(clean_context) > 700:
            clean_context = clean_context[:700] + "..."

        signal_str = ", ".join(signals) if signals else "Subsystem-specific internal bus and logic lines"

        desc_text = f"""[Diagram: {caption}]
- Manual & Reference: {book_name} > {md_name}
- Subsystem / Component: {subsystem}
- Signals & Pinouts / Key Registers: {signal_str}
- Technical Description:
  {clean_context}
- Architectural Significance:
  Detailed hardware specification defining pin assignments, electrical state transitions, bus arbitration, or register encoding for cycle-exact emulation.
"""

        mgr.record_description(img_p, desc_text, root_dir=document_root)
        created_count += 1
        print(f"[{created_count}/{len(unindexed)}] Created sidecar for: {img_p.name}")

    print(f"\nSuccessfully generated {created_count} sidecar descriptions!")
    status_after = mgr.get_unindexed_images(document_root)
    print(f"Remaining unindexed images: {len(status_after)}")


if __name__ == "__main__":
    main()
