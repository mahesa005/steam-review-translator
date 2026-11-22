"""Bootstrap: pastikan paket ``steam_review_translator`` bisa diimpor.

Skrip eksperimen bisa dijalankan tanpa ``pip install -e .``:

    python experiments/task3_summarization/make_mock_inputs.py --help

Modul ini menambahkan ``<repo>/src`` ke ``sys.path`` kalau paketnya belum terinstall.
Kalau paketnya SUDAH terinstall (setup via ``make setup``), tidak ada yang berubah.

Import modul ini harus jadi import pertama di setiap skrip eksperimen.
"""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_on_path() -> None:
    """Tambahkan ``<repo>/src`` ke ``sys.path`` bila paket belum bisa diimpor."""
    try:
        import steam_review_translator  # noqa: F401

        return
    except ModuleNotFoundError:
        pass
    # parents[0] = folder skrip, [1] = experiments, [2] = root repo
    src = Path(__file__).resolve().parents[2] / "src"
    if src.is_dir():
        sys.path.insert(0, str(src))


ensure_repo_on_path()
