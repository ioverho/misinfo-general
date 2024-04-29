import os
from pathlib import Path

import duckdb


def load_db(data_dir: str):
    data_dir = Path(data_dir)

    expected_db_loc = data_dir / "db" / "misinformation_benchmark_metadata.db"

    expected_export_loc = data_dir / "db_export"

    if expected_db_loc.exists():
        return duckdb.connect(str(expected_db_loc))

    elif expected_export_loc.exists():
        if not expected_db_loc.parent.exists():
            os.makedirs(name=expected_db_loc.parent, exist_ok=False)

        db_con = duckdb.connect(str(expected_db_loc))

        db_con.execute(
            f"""
            IMPORT DATABASE '{str(expected_export_loc)}'
            """
        )

        db_con.close()

        return duckdb.connect(str(expected_db_loc))

    else:
        raise ValueError(f"Could not find a db file at: {expected_db_loc}")
