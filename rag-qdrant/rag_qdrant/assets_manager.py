import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from .config import CACHE_FILE
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from rag_qdrant.config import CACHE_FILE


IMAGE_EXTENSIONS = {".svg", ".png", ".jpg", ".jpeg", ".webp"}
IGNORED_DIRS = {".git", ".antigravity", ".venv", "venv", "__pycache__", "node_modules", ".obsidian"}


def compute_file_hash(file_path: Path) -> str:
    """Computes SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class AssetsManager:
    def __init__(self, cache_file: Optional[Path] = None):
        self.cache_file = Path(cache_file) if cache_file else CACHE_FILE

    def load_cache(self) -> Dict[str, Any]:
        """Loads cache data from CACHE_FILE."""
        if self.cache_file.is_file():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"sources": {}, "image_descriptions": {}}

    def save_cache(self, data: Dict[str, Any]):
        """Saves cache data to CACHE_FILE."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def scan_assets(self, root_dir: Path) -> List[Path]:
        """Recursively discovers all diagram/image files under root_dir."""
        root_dir = root_dir.resolve()
        images = []
        for p in root_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                if not any(part in IGNORED_DIRS for part in p.parts):
                    images.append(p)
        return sorted(images)

    def get_sidecar_path(self, image_path: Path) -> Path:
        """Returns standard sidecar path: <image_path>.txt"""
        return image_path.parent / f"{image_path.name}.txt"

    def read_sidecar(self, image_path: Path) -> Optional[str]:
        """Reads sidecar text file if it exists."""
        sidecar = self.get_sidecar_path(image_path)
        if sidecar.is_file():
            try:
                with open(sidecar, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read().strip()
                    if content:
                        return content
            except Exception:
                pass
        # Fallback to without original extension: diagram.txt
        alt_sidecar = image_path.with_suffix(".txt")
        if alt_sidecar.is_file():
            try:
                with open(alt_sidecar, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read().strip()
                    if content:
                        return content
            except Exception:
                pass
        return None

    def get_unindexed_images(self, root_dir: Path) -> List[Dict[str, Any]]:
        """
        Scans all images under root_dir and compares them against image_descriptions in CACHE_FILE.
        Returns a list of image items needing description creation or update.
        """
        cache = self.load_cache()
        img_cache = cache.setdefault("image_descriptions", {})
        images = self.scan_assets(root_dir)

        needed = []
        for img_p in images:
            try:
                rel_path = img_p.relative_to(root_dir).as_posix()
            except ValueError:
                rel_path = img_p.as_posix()

            sidecar_p = self.get_sidecar_path(img_p)
            has_sidecar = sidecar_p.is_file()
            current_hash = compute_file_hash(img_p)

            cached_entry = img_cache.get(rel_path)
            # Support legacy format if cached_entry was a raw string description
            cached_hash = cached_entry.get("hash") if isinstance(cached_entry, dict) else None

            reason = None
            if not cached_hash:
                reason = "missing_from_cache"
            elif cached_hash != current_hash:
                reason = "hash_mismatch"
            elif not has_sidecar:
                reason = "missing_sidecar_file"

            if reason:
                needed.append({
                    "path": img_p,
                    "rel_path": rel_path,
                    "hash": current_hash,
                    "cached_hash": cached_hash,
                    "has_sidecar": has_sidecar,
                    "sidecar_path": sidecar_p,
                    "reason": reason
                })

        return needed

    def record_description(
        self,
        image_path: Path,
        description: str,
        root_dir: Optional[Path] = None
    ) -> Path:
        """Writes sidecar text file and updates CACHE_FILE."""
        sidecar_p = self.get_sidecar_path(image_path)
        sidecar_p.parent.mkdir(parents=True, exist_ok=True)
        with open(sidecar_p, "w", encoding="utf-8") as f:
            f.write(description.strip() + "\n")

        current_hash = compute_file_hash(image_path)
        if root_dir:
            try:
                rel_path = image_path.relative_to(root_dir).as_posix()
            except ValueError:
                rel_path = image_path.as_posix()
        else:
            rel_path = image_path.as_posix()

        cache = self.load_cache()
        img_cache = cache.setdefault("image_descriptions", {})
        img_cache[rel_path] = {
            "hash": current_hash,
            "sidecar_file": str(sidecar_p.name),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.save_cache(cache)
        return sidecar_p


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Manage diagram sidecars and image cache")
    parser.add_argument("path", nargs="?", default=".", help="Root directory containing assets")
    parser.add_argument("--list-unindexed", action="store_true", help="List images needing description")
    parser.add_argument("--status", action="store_true", help="Show asset cache status")
    args = parser.parse_args()

    mgr = AssetsManager()
    target = Path(args.path).resolve()

    if args.status or args.list_unindexed:
        unindexed = mgr.get_unindexed_images(target)
        all_imgs = mgr.scan_assets(target)
        print(f"Target: {target}")
        print(f"Total Images Found : {len(all_imgs)}")
        print(f"Needing Description: {len(unindexed)}")
        if args.list_unindexed:
            for item in unindexed:
                print(f"  [{item['reason']}] {item['rel_path']}")
