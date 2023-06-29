
# Data base practice
from dataclasses import dataclass
from datetime import datetime
import json
import os
from typing import Any, ClassVar, List, Optional, Tuple
import psycopg2
import psycopg2.extras
import psycopg2.extensions

import sqlalchemy
import sqlalchemy.orm
import sqlalchemy.engine

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
    def insert_all(cls, table_name: str, column: str, data: List[Any]) -> None:
        cursor = cls.get_cursor()
        insert_query = f'INSERT INTO {table_name} ({column}) values %s'
        # print(insert_query)
        psycopg2.extras.execute_values (
            cursor, insert_query, new_d, template=None, page_size=100
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
            cursor.execute("SELECT * from stats;")
        else:
            cursor.execute(f"SELECT ({column_names}) from {table_name};")
        return cursor.fetchall()


sql_client = PSQLClient()
sql_client.drop_table("stats")

sql_client.create_table("stats", "id", ["in_game json NOT NULL"])

new_d = [
    (json.dumps(open(filename).read()), )
    for filename in os.scandir("json_data/team_stats/in_game") if filename.is_file()
]
sql_client.insert_all("stats", ("in_game"), new_d)
sql_client.commit()

res = sql_client.select_all("stats")
for r in res:
    id, stats = r
    print("id== ", id)
    print("stats == ", stats)

# cr_str = create_table("stats", "id", ["statistics json NOT NULL"])
# cursor.execute(cr_str)
# conn.commit()

# # Inserts into
# # insert_q =  """ INSERT INTO stats (statistics) VALUES (%s);"""
# # record_to_insert = (json.dumps(valuables),)
# # cursor.execute(insert_q, record_to_insert)
# data = [{"data": 1, "Not Good": 2}, {"Other dict": 3, 2: "four"}]
# new_d = [(json.dumps(d), ) for d in data]


# # print(new_d)
# insert_query = 'insert into stats (statistics) values %s'
# # print(insert_query)
# psycopg2.extras.execute_values (
#     cursor, insert_query, new_d, template=None, page_size=100
# )
# conn.commit()

# for r in result:
#     id_, stats = r
#     print("id ==", id_)
#     print("statistics ==", stats)


# res = cursor.execute("SELECT id from stats;")
# result = cursor.fetchall()
# for r in result:
#     print(r)
# print(res)

# # print("deleting")
# # print("deleting")
# # res = cursor.execute("DROP TABLE stats;")
# # conn.commit()
# # print("deleted")
# # print(res)
# # res = cursor.execute("SELECT * from stats;")
# # print(res)
# # url = sqlalchemy.engine.URL.create(
# #     drivername="postgresql",
# #     username=db_user,
# #     host=db_host,
# #     database=db_name,
# # )

# # print("url", url)
# # engine = sqlalchemy.create_engine(url)
# # connection = engine.connect()

# # # print("engine!", engine)
# # # # print("connected", conn)
# # # # cursor = conn.cursor()
# # metadata = sqlalchemy.MetaData()

# # emp = sqlalchemy.Table(
# #     'stats', metadata,
# #     sqlalchemy.Column("id", sqlalchemy.Integer()),
# #     sqlalchemy.Column("json", sqlalchemy.Integer()),
# #                 )

# # connection.()
# # query = sqlalchemy.insert(emp).values(id=1, json=3)
# # res = connection.execute(query)
# # print(res)

# # res = sqlalchemy.select([emp])
# # print(res)

# # Base = sqlalchemy.orm.declarative_base()


# # Session = sqlalchemy.orm.sessionmaker(bind=engine)

# # session = Session()

