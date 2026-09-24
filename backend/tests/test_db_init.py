from sqlalchemy import create_engine, inspect

from app.db import init_db as init_db_module


def test_init_db_creates_all_core_tables(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    monkeypatch.setattr(init_db_module, "get_engine", lambda: engine)

    init_db_module.init_db()

    assert {"providers", "models", "video_tasks"}.issubset(
        set(inspect(engine).get_table_names())
    )
