import re
from pathlib import Path
from typing import List, Dict, Any

IMAGE_MD_PATTERN = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
IMAGE_WIKI_PATTERN = re.compile(r'!\[\[([^\|\]]+)(?:\|([^\]]+))?\]\]')
HEADER_PATTERN = re.compile(r'^(#{1,4})\s+(.+)$', re.MULTILINE)

class MarkdownChunker:
    def __init__(self, max_chunk_chars: int = 1800, min_chunk_chars: int = 150):
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars

    def extract_images(self, text: str, file_dir: Path) -> List[str]:
        """Extract and resolve image paths referenced in Markdown or Obsidian syntax."""
        image_paths = []
        
        # Markdown standard: ![alt](path)
        for _, path_str in IMAGE_MD_PATTERN.findall(text):
            path_str = path_str.strip()
            # Ignore web links
            if path_str.startswith("http://") or path_str.startswith("https://"):
                continue
            resolved = (file_dir / path_str).resolve()
            if resolved.is_file():
                image_paths.append(str(resolved))

        # Obsidian syntax: ![[image.png]]
        for img_name, _ in IMAGE_WIKI_PATTERN.findall(text):
            img_name = img_name.strip()
            # Check in same directory or common subdirectories (attachments, images, assets)
            candidates = [
                file_dir / img_name,
                file_dir / "attachments" / img_name,
                file_dir / "images" / img_name,
                file_dir / "assets" / img_name,
            ]
            for cand in candidates:
                if cand.is_file():
                    image_paths.append(str(cand.resolve()))
                    break
                    
        return list(set(image_paths))

    def chunk_markdown(self, file_path: Path, root_dir: Path, source: str) -> List[Dict[str, Any]]:
        """Splits markdown file into hierarchical chunks with metadata."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as e:
            print(f"[Warning] Failed to read {file_path}: {e}")
            return []

        if not content.strip():
            return []

        file_dir = file_path.parent
        rel_path = file_path.relative_to(root_dir).as_posix()

        # Split by markdown headers
        lines = content.splitlines(keepends=True)
        sections = []
        current_header = "Intro"
        current_lines = []
        header_stack = {}

        in_code_block = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code_block = not in_code_block

            header_match = HEADER_PATTERN.match(line) if not in_code_block else None
            if header_match:
                # Save previous section
                if current_lines:
                    sec_text = "".join(current_lines).strip()
                    if sec_text:
                        sections.append({
                            "header": current_header,
                            "text": sec_text
                        })
                    current_lines = []

                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                header_stack[level] = title
                # Clear deeper levels
                for k in list(header_stack.keys()):
                    if k > level:
                        del header_stack[k]
                
                # Build breadcrumbs e.g. "Hardware > Copper > Timing"
                current_header = " > ".join([header_stack[k] for k in sorted(header_stack.keys())])
                current_lines.append(line)
            else:
                current_lines.append(line)

        # Append last section
        if current_lines:
            sec_text = "".join(current_lines).strip()
            if sec_text:
                sections.append({
                    "header": current_header,
                    "text": sec_text
                })

        # Merge small sections, frontmatter & sparse title-only blocks
        merged_sections = []
        buffer_text = ""

        for sec in sections:
            sec_text = sec["text"]
            header = sec["header"]

            if buffer_text:
                sec_text = buffer_text + "\n\n" + sec_text
                buffer_text = ""

            # Check substantive body (strip markdown headers, frontmatter, wiki links)
            cleaned = re.sub(r'^---[\s\S]*?---\s*', '', sec_text.strip())
            cleaned = re.sub(r'^#{1,6}\s+.*$', '', cleaned, flags=re.MULTILINE)
            cleaned = re.sub(r'>\s*See also:.*$', '', cleaned, flags=re.MULTILINE)
            cleaned = re.sub(r'\[\[.*?\]\]', '', cleaned)
            substantive_body = cleaned.strip()

            has_image = bool(self.extract_images(sec_text, file_dir))
            has_code = "```" in sec_text

            if len(substantive_body) < 140 and not has_image and not has_code:
                buffer_text = sec_text
            else:
                merged_sections.append({"header": header, "text": sec_text})

        if buffer_text:
            if merged_sections:
                last = merged_sections[-1]
                if len(last["text"]) + len(buffer_text) + 2 <= self.max_chunk_chars:
                    last["text"] = last["text"] + "\n\n" + buffer_text
                else:
                    merged_sections.append({"header": sections[-1]["header"] if sections else "Note", "text": buffer_text})
            else:
                merged_sections.append({"header": "Note", "text": buffer_text})

        chunks = []
        chunk_idx = 0

        for sec in merged_sections:
            sec_text = sec["text"]
            header = sec["header"]

            # If section text is within max size, keep it as single chunk
            if len(sec_text) <= self.max_chunk_chars:
                images = self.extract_images(sec_text, file_dir)
                chunks.append({
                    "chunk_id": f"{rel_path}#{chunk_idx}",
                    "file_path": str(file_path.resolve()),
                    "relative_path": rel_path,
                    "source": source,
                    "header": header,
                    "content": sec_text,
                    "images": images,
                    "chunk_index": chunk_idx
                })
                chunk_idx += 1
            else:
                # Sub-chunk larger sections by paragraphs
                paragraphs = re.split(r'\n\s*\n', sec_text)
                sub_buffer = []
                sub_len = 0

                for para in paragraphs:
                    if sub_len + len(para) > self.max_chunk_chars and sub_buffer:
                        combined = "\n\n".join(sub_buffer).strip()
                        images = self.extract_images(combined, file_dir)
                        chunks.append({
                            "chunk_id": f"{rel_path}#{chunk_idx}",
                            "file_path": str(file_path.resolve()),
                            "relative_path": rel_path,
                            "source": source,
                            "header": header,
                            "content": combined,
                            "images": images,
                            "chunk_index": chunk_idx
                        })
                        chunk_idx += 1
                        sub_buffer = [para]
                        sub_len = len(para)
                    else:
                        sub_buffer.append(para)
                        sub_len += len(para)

                if sub_buffer:
                    combined = "\n\n".join(sub_buffer).strip()
                    # If trailing paragraph is small, merge into previous chunk if within 125% limit
                    if len(combined) < 140 and chunks and chunks[-1]["relative_path"] == rel_path and chunks[-1]["header"] == header:
                        if len(chunks[-1]["content"]) + len(combined) <= int(self.max_chunk_chars * 1.25):
                            chunks[-1]["content"] += "\n\n" + combined
                            chunks[-1]["images"] = list(set(chunks[-1]["images"] + self.extract_images(combined, file_dir)))
                            continue

                    images = self.extract_images(combined, file_dir)
                    chunks.append({
                        "chunk_id": f"{rel_path}#{chunk_idx}",
                        "file_path": str(file_path.resolve()),
                        "relative_path": rel_path,
                        "source": source,
                        "header": header,
                        "content": combined,
                        "images": images,
                        "chunk_index": chunk_idx
                    })
                    chunk_idx += 1

        return chunks
