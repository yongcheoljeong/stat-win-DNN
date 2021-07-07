import pymysql
from sqlalchemy import create_engine 
import pandas as pd 
import mysql_auth

class MySQLConnection():
    def __init__(self, input_df=None, DB_info=mysql_auth.NYXLDB_FinalStat):
        if input_df is None: 
            pass 
        
        dbname = DB_info['dbname']
        hostname = DB_info['hostname']
        username = DB_info['username']
        pwd = DB_info['pwd']
        port = DB_info['port']
        # define input_df
        self.input_df = input_df
        # dbname
        self.dbname = dbname
        # create engine
        self.engine = create_engine('mysql+pymysql://' + username + ':' + pwd + '@' + hostname + ':' + str(port) + '/' + dbname , echo=False)

    def export_to_db(self, table_name, if_exists='replace'):
        table_name = table_name.lower() # MySQL DB에서 table 이름을 자동으로 소문자로 바꿔주기 때문에 'replace' 기능 쓰려면 필수
        self.input_df.to_sql(name=table_name, con=self.engine, schema=self.dbname, if_exists=if_exists) # if_exsits:{'fail', 'replace', 'append'}

    def get_table_names(self):
        table_names = self.engine.table_names()
        
        return table_names

    def read_table_as_df(self, table_name):
        table_df = pd.read_sql(
            sql=f"SELECT * FROM `{table_name}`",
            con=self.engine,
        )
        if 'index' in table_df.columns.tolist():
            table_df.drop(columns='index', inplace=True) # drop 'index' column
        elif 'level_0' in table_df.columns.tolist():
            table_df.drop(columns='level_0', inplace=True) # drop 'level_0 column

        return table_df

    def read_table_with_sql(self, sql):
        table_df = pd.read_sql(
            sql=sql,
            con=self.engine
        )
        if 'index' in table_df.columns.tolist():
            table_df.drop(columns='index', inplace=True) # drop 'index' column
        elif 'level_0' in table_df.columns.tolist():
            table_df.drop(columns='level_0', inplace=True) # drop 'level_0 column

        return table_df

    def read_all_tables_as_df(self):
        sql_union = ''
        table_names = self.get_table_names()
        for tablename in table_names:
            if tablename is table_names[-1]:
                sql = f"""
                SELECT * from `{tablename}`;
                """
                sql_union = sql_union + sql 
            else:
                sql = f"""
                SELECT * from `{tablename}`
                """
                sql_union = sql_union + sql + ' UNION ALL '
        
        table_df = self.read_table_with_sql(sql_union)
        return table_df