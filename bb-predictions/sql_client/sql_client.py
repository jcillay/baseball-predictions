import json
from typing import Any, ClassVar, List, Optional, Tuple
import psycopg2
import psycopg2.extras
import psycopg2.extensions


class PSQLClient:
    """"""
    cursor: ClassVar[Optional[psycopg2.extensions.cursor]] = None
    db_conn: ClassVar[Optional[psycopg2.extensions.connection]] = None

    @classmethod
    def get_db_connection(cls) -> psycopg2.extensions.connection:
        if cls.db_conn is None:
            with open("configs.json") as f:
                valuables = json.load(f)

            db_name = valuables["db_name"]
            db_host = valuables["db_host"]
            db_port = valuables["db_port"]
            db_user = valuables["db_user"]
            db_pass = valuables["db_password"]
            cls.db_conn = psycopg2.connect(
                database=db_name,
                host=db_host,
                user=db_user,
                password=db_pass,
                port=db_port
            )
        return cls.db_conn

    @classmethod
    def get_cursor(cls) -> psycopg2.extensions.cursor:
        if cls.cursor is None:
            conn = cls.get_db_connection()
            cls.cursor = conn.cursor()
        return cls.cursor

    @classmethod
    def delete_data(cls, table_name: str, game_id: str) -> None:
        """ Drops a db table if the table exists"""
        cur = cls.get_cursor()
        cur.execute(f"DELETE FROM {table_name} WHERE game_id = %s;", (game_id,))

    @classmethod
    def drop_table(cls, table_name: str) -> None:
        """ Drops a db table if the table exists"""
        cur = cls.get_cursor()
        cur.execute("select exists(select * from information_schema.tables where table_name=%s)", (table_name,))
        fetched = cur.fetchone()
        if fetched is not None and fetched[0]:
            print(f"Dropping table {table_name}.")
            cur.execute(f"DROP TABLE {table_name}")

    @classmethod
    def create_table(cls, table_name: str, primary_key: str,
                    columns: List[str]) -> None:
        """ Creates a db table if the table does not already exist. """
        cursor = cls.get_cursor()
        creation_string = f"CREATE TABLE IF NOT EXISTS {table_name} ( " \
            f"\n {primary_key} SERIAL PRIMARY KEY,"
        for i in range(len(columns)):
            creation_string += "\n " + columns[i]
            creation_string += "," if i < len(columns) - 1 else ""

        creation_string += "\n)"
        print("creating table with creation string", creation_string)
        cursor.execute(creation_string)

    @classmethod
    def insert_all(cls, table_name: str, column_names: Tuple[str], data: List[Any]) -> None:
        cursor = cls.get_cursor()
        insert_query = f'INSERT INTO {table_name} ({",".join(column_names)}) values %s'
        psycopg2.extras.execute_values (
            cursor, insert_query, data, template=None, page_size=100
        )

    @classmethod
    def commit(cls) -> None:
        db_conn = cls.get_db_connection()
        db_conn.commit()

    @classmethod
    def select_all(cls, table_name: str, column_names: Optional[Tuple[str]] = None) -> List[Tuple[Any]]:
        "Either selects every entry in the db or every entry for a specific column. "
        cursor = cls.get_cursor()
        if column_names is None:
            cursor.execute(f"SELECT * from {table_name};")
        else:
            cursor.execute(f"SELECT ({column_names}) from {table_name};")
        return cursor.fetchall()

    @classmethod
    def select_by_columns(cls, table_name: str, column_conditions: List[str]) -> List[Tuple[Any]]:
        "Either selects every entry in the db or every entry for a specific column. "
        cursor = cls.get_cursor()
        select_stmt = f"SELECT * from {table_name} WHERE "
        for i in range(len(column_conditions)):
            select_stmt += column_conditions[i]
            select_stmt += " AND " if i < len(column_conditions) - 1 else ";"
        cursor.execute(select_stmt)
        return cursor.fetchall()

    @classmethod
    def select_column(cls, table_name: str, columns: List[str]) -> List[Tuple]:
        "Either selects every entry in the db or every entry for a specific column. "
        cursor = cls.get_cursor()
        select_stmt = f"SELECT "
        for i in range(len(columns)):
            select_stmt += columns[i]
        select_stmt += f" FROM {table_name};"
        print("select_stmt", select_stmt)
        cursor.execute(select_stmt)
        return cursor.fetchall()
