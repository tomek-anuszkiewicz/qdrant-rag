import hashlib
import threading
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Dict, List, Any
from PIL import Image

logging.getLogger("google_genai").setLevel(logging.ERROR)
try:
    from google.genai.models import Models, AsyncModels
    Models._logged_afc_warning = True
    AsyncModels._logged_afc_warning = True
except Exception:
    pass

warnings.filterwarnings("ignore", message=".*automatic function calling.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*genai.*")

from .config import GEMINI_API_KEY, VISION_MODEL
from .assets_manager import AssetsManager


class VisionAnalyzer:
    """Resolves diagram descriptions via local sidecar text files (<image>.txt) with optional cloud fallback."""

    def __init__(self, cache: Dict[str, Any]):
        self.cache = cache
        self.lock = threading.Lock()
        self.assets_mgr = AssetsManager()
        self.client = None
        self.legacy_model = None
        self.cloud_available = bool(GEMINI_API_KEY)
        self.available = True  # Always available because local sidecars require no cloud API
        self._init_cloud_client()

    def _init_cloud_client(self):
        if not GEMINI_API_KEY:
            self.cloud_available = False
            return
        try:
            from google import genai
            self.client = genai.Client(api_key=GEMINI_API_KEY)
            self.cloud_available = True
        except Exception:
            try:
                import google.generativeai as genai_legacy
                genai_legacy.configure(api_key=GEMINI_API_KEY)
                self.legacy_model = genai_legacy.GenerativeModel(VISION_MODEL)
                self.cloud_available = True
            except Exception:
                self.client = None
                self.cloud_available = False

    def _file_hash(self, file_path: Path) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def analyze_image(self, image_path_str: str) -> Optional[str]:
        image_path = Path(image_path_str)
        if not image_path.is_file():
            return None

        # 1. Check local sidecar file (<image>.txt or <image_without_ext>.txt)
        sidecar_text = self.assets_mgr.read_sidecar(image_path)
        if sidecar_text:
            return sidecar_text

        # 2. Check in-memory cache
        try:
            img_hash = self._file_hash(image_path)
            with self.lock:
                if img_hash in self.cache:
                    cached_val = self.cache[img_hash]
                    if isinstance(cached_val, str):
                        return cached_val
                    elif isinstance(cached_val, dict) and "description" in cached_val:
                        return cached_val["description"]
        except Exception:
            return None

        # 3. Optional cloud fallback if API key is active and cloud is available
        if not self.cloud_available or (not self.client and not self.legacy_model):
            return None

        prompt = (
            "Analyze this technical image, diagram, or schematic. Provide a concise, highly factual "
            "technical description of what is depicted (architecture, register fields, circuit, timing, or block diagram) "
            "and extract all visible text, labels, values, and identifiers (OCR)."
        )

        description = ""
        try:
            pil_img = Image.open(image_path)
            if self.client:
                response = self.client.models.generate_content(
                    model=VISION_MODEL,
                    contents=[pil_img, prompt]
                )
                description = response.text or ""
            elif hasattr(self, 'legacy_model') and self.legacy_model:
                response = self.legacy_model.generate_content([prompt, pil_img])
                description = response.text or ""
        except Exception as e:
            err_str = str(e)
            with self.lock:
                if self.cloud_available and ("RESOURCE_EXHAUSTED" in err_str or "429" in err_str or "NOT_FOUND" in err_str or "404" in err_str):
                    self.cloud_available = False
            return None

        description = description.strip()
        if description:
            with self.lock:
                self.cache[img_hash] = description
            # Write sidecar to disk so it's permanently cached locally
            try:
                self.assets_mgr.record_description(image_path, description)
            except Exception:
                pass
            return description

        return None

    def analyze_images_parallel(
        self,
        image_paths: List[str],
        max_workers: int = 8,
        progress_cb=None
    ) -> Dict[str, str]:
        """Resolves diagram descriptions concurrently from sidecars or cache."""
        if not image_paths:
            return {}

        unique_paths = list({str(Path(p).resolve()) for p in image_paths if Path(p).is_file()})
        results = {}
        total = len(unique_paths)
        completed = 0

        def _task(img_p):
            return img_p, self.analyze_image(img_p)

        workers = min(max_workers, len(unique_paths))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            try:
                future_to_path = {executor.submit(_task, p): p for p in unique_paths}
                for future in as_completed(future_to_path):
                    completed += 1
                    try:
                        p, desc = future.result()
                        if desc:
                            results[p] = desc
                        if progress_cb:
                            progress_cb(completed, total, Path(p).name)
                    except Exception:
                        pass
            except KeyboardInterrupt:
                executor.shutdown(wait=False, cancel_futures=True)
                raise

        return results
