import os
from typing import Any
import logging
import sqlite3
from itertools import repeat

import pandas as pd
from pydantic import BaseModel

import tensorflow_datasets as tfds

logger = logging.getLogger()


class DataLoader(BaseModel):
    db_path: str = 'data/movielens.db'
    force_rebuild_db: bool = False

    def __init__(self, **data):
        super().__init__(**data)
        if not os.path.isfile(self.db_path) or self.force_rebuild_db:
            self.init_db_from_tfds()

    def init_db_from_tfds(self):
        logger.info('Initializing database from tensorflow dataset.')
        movies, movies_info = tfds.load(name='movielens/100k-movies', with_info=True)
        ratings, ratings_info = tfds.load(name='movielens/100k-ratings', with_info=True)
        df_movies = tfds.as_dataframe(ds=movies['train'], ds_info=movies_info)
        df_movies['movie_id'] = df_movies['movie_id'].astype(int)
        df_movies['movie_title'] = df_movies['movie_title'].astype(str)

        logger.info('Created movies table.')

        df_genres = pd.DataFrame({'genre_id': [i for i in range(len(movies_info.features['movie_genres'].names))],
                                  'genre_name': movies_info.features['movie_genres'].names})

        logger.info('Created genres table.')

        df_users_ratings = tfds.as_dataframe(ds=ratings['train'], ds_info=ratings_info).drop(columns=['movie_genres',
                                                                                                      'movie_title'])
        df_users_ratings['movie_id'] = df_users_ratings['movie_id'].astype(int)
        df_users_ratings['user_id'] = df_users_ratings['user_id'].astype(int)
        df_users_ratings['user_occupation_text'] = df_users_ratings['user_occupation_text'].astype(str)
        df_users_ratings['user_zip_code'] = df_users_ratings['user_zip_code'].astype(str)

        df_users = df_users_ratings.loc[:, ['user_id', 'bucketized_user_age', 'raw_user_age', 'user_gender',
                                            'user_occupation_label', 'user_occupation_text',
                                            'user_zip_code']].drop_duplicates()
        df_users['hashed_passwords'] = ''

        logger.info('Created users table.')

        df_ratings = df_users_ratings.loc[:, ['user_id', 'movie_id', 'user_rating']].drop_duplicates()
        df_ratings['rating_id'] = df_ratings.index

        logger.info('Created ratings table.')

        if not os.path.isdir(self.db_path.split('/')[0]):
            os.mkdir(self.db_path.split('/')[0])

        logger.info('Writing tables to local sqlite database.')
        with sqlite3.connect(self.db_path) as conn:
            df_movies.to_sql('movies', conn, if_exists='replace', index=False)
            df_genres.to_sql('genres', conn, if_exists='replace', index=False)
            df_users.to_sql('users', conn, if_exists='replace', index=False)
            df_ratings.to_sql('ratings', conn, if_exists='replace', index=False)

        logger.info('Database initialization finished.')

    def query_data(self, query: str, params: list[Any] | None = None):
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql(query, conn, params=params)

    def get_full_table(self, table: str) -> pd.DataFrame:
        return self.query_data(query=f'select * from {table}')

    def get_table_partial(self, table: str, percentage: float):
        n_rows = self.query_data(query=f'select count(*) from {table};').values[0][0] * percentage
        return self.query_data(query=f'select * from {table} limit {n_rows};')

    def random_sample_table(self, table: str, n: int):
        return self.query_data(query=f'select * from {table} order by random() limit {n};')

    def query_on_col_value(self, table: str, col_name: str, col_value: Any):
        sql = f'select * from {table} where {col_name}=?'
        return self.query_data(query=sql, params=[col_value])

    def insert_data(self, table: str, data: dict[str, Any]):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            sql = f'insert into {table}({", ".join(list(data.keys()))}) values({",".join(repeat("?", len(data)))})'
            cur.execute(sql, tuple(data.values()))

    def update_data(self, table: str, id_tuple: tuple[str, Any], data: dict[str, Any]):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            sql = f'update {table} set {"= ? ,".join(list(data.keys()))} = ? where {id_tuple[0]} = ?'
            cur.execute(sql, tuple(data.values()) + (id_tuple[1],))

    def delete_data(self, table: str, id_tuple: tuple[str, Any]):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            sql = f'DELETE FROM {table} WHERE {id_tuple[0]}=?'
            cur.execute(sql, (id_tuple[1],))
