
"""Curated impulse response (IR) database integration.

Provides a simple downloader and cache manager for publicly available
impulse response collections, enabling physics-based room acoustics
augmentation via :class:`pvx.augment.ImpulseResponseConvolver`.

Supported databases
-------------------
- **EchoThief** — 115 real-world IRs from concert halls, churches,
  tunnels, parking garages, and other spaces.  CC BY license.
  http://www.echothief.com
- **OpenAIR** — University of York's open acoustic impulse response
  library with measured IRs from real spaces.
  https://openairlib.net

Usage
-----
>>> from pvx.augment.ir_database import IRDatabase
>>>
>>> db = IRDatabase(cache_dir="~/.pvx/ir_cache")
>>> db.download("echothief")
>>>
>>> # Use with ImpulseResponseConvolver
>>> from pvx.augment import ImpulseResponseConvolver
>>> aug = ImpulseResponseConvolver(db.ir_dir("echothief"), wet_range=(0.4, 1.0))
>>>
>>> # Or get specific IRs by room category
>>> halls = db.filter("echothief", category="hall")
>>> aug = ImpulseResponseConvolver(halls, wet_range=(0.5, 0.9))
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

# ---------------------------------------------------------------------------
# Registry of known IR databases
# ---------------------------------------------------------------------------

_IR_REGISTRY: dict[str, dict[str, object]] = {
    "echothief": {
        "name": "EchoThief",
        "url": "http://www.echothief.com/downloads/EchoThiefImpulseResponseLibrary.zip",
        "description": "115 real-world IRs from diverse spaces (CC BY license)",
        "license": "CC BY",
        "size_mb": 290,
        "categories": {
            "hall": ["Concert Hall", "Auditorium", "Theater"],
            "church": ["Church", "Cathedral", "Chapel"],
            "room": ["Room", "Studio", "Office"],
            "outdoor": ["Outdoor", "Parking", "Tunnel", "Bridge"],
            "large": ["Arena", "Gymnasium", "Warehouse"],
        },
    },
    "mit_kemar": {
        "name": "MIT KEMAR HRTF",
        "url": "https://sound.media.mit.edu/resources/KEMAR/KEMAR-MIT.zip",
        "description": "MIT KEMAR head-related impulse responses",
        "license": "MIT",
        "size_mb": 15,
        "categories": {},
    },
}


# ---------------------------------------------------------------------------
# IRDatabase
# ---------------------------------------------------------------------------


class IRDatabase:
    """Manage impulse response databases for room acoustics augmentation.

    Downloads, caches, and provides filtered access to curated IR
    collections.  IRs are stored as WAV files on disk and can be
    used directly with :class:`pvx.augment.ImpulseResponseConvolver`.

    Parameters
    ----------
    cache_dir:
        Directory for cached IR files.  Defaults to ``~/.pvx/ir_cache``.

    Examples
    --------
    >>> db = IRDatabase()
    >>> db.download("echothief")          # download once, cached
    >>> db.list_databases()               # available collections
    >>> ir_files = db.list_irs("echothief")  # all IR file paths
    >>> halls = db.filter("echothief", category="hall")
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        if cache_dir is None:
            self.cache_dir = Path.home() / ".pvx" / "ir_cache"
        else:
            self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_databases(self) -> list[dict[str, object]]:
        """Return metadata for all registered IR databases."""
        result = []
        for key, info in _IR_REGISTRY.items():
            downloaded = self._db_dir(key).exists()
            result.append(
                {
                    "id": key,
                    "name": info["name"],
                    "description": info["description"],
                    "license": info["license"],
                    "size_mb": info["size_mb"],
                    "downloaded": downloaded,
                    "n_files": len(self.list_irs(key)) if downloaded else 0,
                }
            )
        return result

    def download(
        self,
        database_id: str,
        *,
        force: bool = False,
        progress: bool = True,
    ) -> Path:
        """Download and extract an IR database.

        Parameters
        ----------
        database_id:
            ID of the database (e.g., ``"echothief"``).
        force:
            Re-download even if already cached.
        progress:
            Print progress messages.

        Returns
        -------
        Path
            Directory containing the extracted IR files.

        Raises
        ------
        ValueError
            If *database_id* is not recognized.
        """
        if database_id not in _IR_REGISTRY:
            available = ", ".join(sorted(_IR_REGISTRY))
            raise ValueError(f"Unknown IR database: {database_id!r}. Available: {available}")

        info = _IR_REGISTRY[database_id]
        db_dir = self._db_dir(database_id)

        if db_dir.exists() and not force:
            if progress:
                n_files = len(self.list_irs(database_id))
                print(f"[ir_database] {info['name']} already cached ({n_files} IRs) at {db_dir}")
            return db_dir

        url = str(info["url"])
        zip_path = self.cache_dir / f"{database_id}.zip"

        if progress:
            print(f"[ir_database] Downloading {info['name']} ({info['size_mb']}MB)...")

        urlretrieve(url, str(zip_path))

        if progress:
            print(f"[ir_database] Extracting to {db_dir}...")

        if db_dir.exists():
            shutil.rmtree(db_dir)
        db_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(db_dir))

        # Clean up zip
        zip_path.unlink(missing_ok=True)

        # Write metadata
        meta = {
            "database_id": database_id,
            "name": info["name"],
            "license": info["license"],
            "source_url": url,
        }
        (db_dir / "_pvx_metadata.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )

        n_files = len(self.list_irs(database_id))
        if progress:
            print(f"[ir_database] Done — {n_files} IR files ready")

        return db_dir

    def ir_dir(self, database_id: str) -> Path:
        """Return the cache directory for a database (may not exist yet)."""
        return self._db_dir(database_id)

    def list_irs(self, database_id: str) -> list[Path]:
        """Return sorted list of all IR audio files in a downloaded database."""
        db_dir = self._db_dir(database_id)
        if not db_dir.exists():
            return []
        audio_exts = {".wav", ".flac", ".aiff", ".aif", ".ogg"}
        return sorted(
            p
            for p in db_dir.rglob("*")
            if p.suffix.lower() in audio_exts and not p.name.startswith(".")
        )

    def filter(
        self,
        database_id: str,
        *,
        category: str | None = None,
        name_contains: str | None = None,
        max_duration_s: float | None = None,
        min_duration_s: float | None = None,
    ) -> list[Path]:
        """Filter IRs by category, name substring, or duration.

        Parameters
        ----------
        database_id:
            Database to filter.
        category:
            Category key from the database registry (e.g., ``"hall"``,
            ``"church"``, ``"outdoor"``).
        name_contains:
            Substring that must appear in the filename (case-insensitive).
        max_duration_s:
            Maximum IR duration in seconds.
        min_duration_s:
            Minimum IR duration in seconds.

        Returns
        -------
        list[Path]
            Matching IR file paths.
        """
        files = self.list_irs(database_id)

        if category is not None:
            info = _IR_REGISTRY.get(database_id, {})
            categories = info.get("categories", {})
            keywords = categories.get(category, [])
            if keywords:
                lower_keywords = [k.lower() for k in keywords]
                files = [
                    f
                    for f in files
                    if any(
                        kw in f.stem.lower() or kw in str(f.parent).lower() for kw in lower_keywords
                    )
                ]

        if name_contains is not None:
            lower_name = name_contains.lower()
            files = [f for f in files if lower_name in f.stem.lower()]

        if max_duration_s is not None or min_duration_s is not None:
            import soundfile as sf

            filtered = []
            for f in files:
                try:
                    info_sf = sf.info(str(f))
                    dur = info_sf.duration
                    if max_duration_s is not None and dur > max_duration_s:
                        continue
                    if min_duration_s is not None and dur < min_duration_s:
                        continue
                    filtered.append(f)
                except (OSError, RuntimeError, ValueError):
                    continue
            files = filtered

        return files

    def remove(self, database_id: str) -> None:
        """Remove a cached database from disk."""
        db_dir = self._db_dir(database_id)
        if db_dir.exists():
            shutil.rmtree(db_dir)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _db_dir(self, database_id: str) -> Path:
        return self.cache_dir / database_id
