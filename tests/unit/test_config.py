import os
from pathlib import Path

import pytest

from steam_review_translator.aspect.mapping import MappingParams, load_mapping_params, map_topic
from steam_review_translator.core.config import (
    PipelineConfig,
    get_path,
    load_dotenv,
    load_pipeline_config,
    load_yaml,
    project_root,
)
from steam_review_translator.core.logging import ROOT_LOGGER, get_logger
from steam_review_translator.core.schemas import Aspect


def test_project_root_is_repo() -> None:
    root = project_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "configs" / "pipeline.yaml").is_file()


def test_project_root_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SRT_PROJECT_ROOT", str(tmp_path))
    assert project_root() == tmp_path.resolve()


def test_load_yaml(tmp_path: Path) -> None:
    f = tmp_path / "x.yaml"
    f.write_text("a: 1\nb: [x, y]\n", encoding="utf-8")
    assert load_yaml(f) == {"a": 1, "b": ["x", "y"]}


def test_load_yaml_empty_and_invalid(tmp_path: Path) -> None:
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    assert load_yaml(empty) == {}
    bad = tmp_path / "list.yaml"
    bad.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_yaml(bad)


def test_load_yaml_relative_to_root() -> None:
    assert "aspects" in load_yaml("configs/task2/lexicon.yaml")


def test_get_path_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.delenv("ARTIFACTS_DIR", raising=False)
    root = project_root()
    assert get_path("data") == root / "data"
    assert get_path("raw") == root / "data" / "raw"
    assert get_path("logs") == root / "artifacts" / "logs"


def test_get_path_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    assert get_path("processed") == tmp_path / "processed"
    monkeypatch.setenv("DATA_DIR", "./elsewhere")
    assert get_path("data") == project_root() / "elsewhere"


def test_get_path_unknown() -> None:
    with pytest.raises(KeyError):
        get_path("nope")


def test_load_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "# comment\nSRT_TEST_A=1\nexport SRT_TEST_B='two'\nSRT_TEST_C=new\n\nbroken\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("SRT_TEST_A", raising=False)
    monkeypatch.delenv("SRT_TEST_B", raising=False)
    monkeypatch.setenv("SRT_TEST_C", "existing")
    parsed = load_dotenv(env)
    assert parsed == {"SRT_TEST_A": "1", "SRT_TEST_B": "two", "SRT_TEST_C": "new"}
    assert os.environ["SRT_TEST_A"] == "1"
    assert os.environ["SRT_TEST_B"] == "two"
    assert os.environ["SRT_TEST_C"] == "existing"  # not overridden
    assert load_dotenv(tmp_path / "missing.env") == {}


def test_pipeline_config_defaults_to_mocks() -> None:
    cfg = load_pipeline_config()
    assert (cfg.sentiment.impl, cfg.aspect.impl, cfg.summarization.impl) == ("mock",) * 3
    assert PipelineConfig().aspect.impl == "mock"


def test_pipeline_config_rejects_unknown_keys(tmp_path: Path) -> None:
    f = tmp_path / "p.yaml"
    f.write_text("sentiment: {impl: mock}\ntranslator: {impl: x}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_pipeline_config(f)


def test_mapping_params() -> None:
    assert load_mapping_params() == MappingParams(floor=0.3, ratio=0.5)
    with pytest.raises(ValueError):
        MappingParams(floor=1.5)
    p = MappingParams()
    assert map_topic({Aspect.STORY: 0.8, Aspect.AUDIO: 0.1}, p) is Aspect.STORY
    assert map_topic({Aspect.STORY: 0.2}, p) is Aspect.OTHER  # below floor
    assert map_topic({Aspect.STORY: 0.4, Aspect.AUDIO: 0.4, Aspect.PRICE: 0.4}, p) is Aspect.OTHER
    assert map_topic({}, p) is Aspect.OTHER


def test_get_logger_is_idempotent() -> None:
    import logging

    a = get_logger("x")
    b = get_logger("steam_review_translator.x")
    assert a is b and a.name == f"{ROOT_LOGGER}.x"
    n = len(logging.getLogger(ROOT_LOGGER).handlers)
    get_logger("y")
    assert len(logging.getLogger(ROOT_LOGGER).handlers) == n
    assert not a.handlers  # children propagate to the package root logger
