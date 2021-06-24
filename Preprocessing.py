import pandas as pd 
import numpy as np
from MySQLConnection import *
from tqdm import tqdm 
import FeatureSelection
import FeatureEngineering

class Preprocessing:
    def __init__(self, DB_info):
        self.ScrimLog = MySQLConnection(DB_info=DB_info)
        self.DB_table_names = self.ScrimLog.get_table_names()

    def import_data(self):
        # make sql here 
        sql_union = ''
        tablenames = self.DB_table_names #[x for x in self.DB_table_names if (x.startswith('202105')) or (x.startswith('202106'))]
        for tablename in tablenames: 
            if tablename is self.DB_table_names[-1]:
                sql = f"""
                SELECT `MatchId`, `Map`, `Section`, `Timestamp`, `Team`, `Player`, `Hero`, 
                `UltimateCharge`, `UltimatesUsed/s`, `TF_order`, `TF_winner`, `TF_RCP_sum`, `RCP`, `Position` from `{tablename}`;
                """
                sql_union = sql_union + sql 
            else:
                sql = f"""
                SELECT `MatchId`, `Map`, `Section`, `Timestamp`, `Team`, `Player`, `Hero`, 
                `UltimateCharge`, `UltimatesUsed/s`, `TF_order`, `TF_winner`, `TF_RCP_sum`, `RCP`, `Position` from `{tablename}`
                """
                sql_union = sql_union + sql + ' UNION '
        print(sql_union)
        # read from db
        self.df_all = self.ScrimLog.read_table_with_sql(sql_union)
    
    def select_features(self):
        self.import_data()
        self.idx = FeatureSelection.selected_index
        self.selected_features = FeatureSelection.selected_features
        output_df = self.df_all[self.idx + self.selected_features].set_index(self.idx)
        return output_df

    def engineer_features(self):
        input_df = self.select_features()
        # Position
        input_df = FeatureEngineering.PositionInfo(input_df)

        # SkillUsed Timestamp
        # input_df = FeatureEngineering.SkillUsedTimestampInfo(input_df)

        # Categorical Encoding
        input_df = FeatureEngineering.CategoricalEncoding(input_df)

        # Scaling
        # input_df = FeatureEngineering.scale_data(input_df)

        output_df = input_df 
        return output_df
    
    def get_data_table_from_DB(self):
        output_df = self.engineer_features()
        return output_df
    
    def get_data_table_from_file(self, filename):
        output_df = pd.read_csv(filename)
        return output_df
    
    def get_input_data_LSTM(self, input_df, time_shift=10, features=['Position_x', 'Position_y', 'Position_z'], target=['RCP']):
        df_X = pd.DataFrame()
        df_y = pd.DataFrame()
        # MatchId, Map, Section, Team, Player로 grouping하고 sorting해서 10초간 데이터끼리 매칭
        hero_col = [x for x in input_df.columns if x.startswith('Hero')]
        df_init = input_df[hero_col+features+target]
        
        df_merge = df_init
        for s in tqdm(range(1, time_shift)):
            df_shift = df_init.groupby(['MatchId', 'Map', 'Section', 'Team', 'Player']).shift(-s)
            df_merge = pd.merge(df_merge, df_shift, how='inner', left_index=True, right_index=True, suffixes=('', f'_{s}'), copy=False)
            del(df_shift)
        df_merge.dropna(inplace=True)
        df_pivot = df_merge.pivot_table(index=['MatchId', 'Map', 'Section', 'Timestamp'], columns=['Team', 'Player'])
        df_pivot.dropna(inplace=True)
        col_names = df_pivot.columns.get_level_values(0)
        df_X = df_pivot.drop(columns=[x for x in col_names if x.startswith(target[0])], level=0)
        df_y = df_pivot[target].mean(axis=1)

        print('#nan in df_X: ', df_X.isna().sum().sum())
        print('#nan in df_y: ', df_y.isna().sum().sum())
        print('#nan in df_X: ', df_X.isna().sum())
        print('#nan in df_y: ', df_y.isna().sum())
        self.df_X = df_X 
        self.df_y = df_y
        X = df_X.values
        y = df_y.values
        print(f'X shape: {X.shape}')
        print(f'y shape: {y.shape}')

        self.X = X 
        self.y = y

        return X, y

    def get_input_data_DNN_classification(self, input_df, features=['Position_x', 'Position_y', 'Position_z'], target=['TF_winner']):
        df_X = pd.DataFrame()
        df_y = pd.DataFrame()
        input_df.dropna(axis=1, how='all', inplace=True) # Player 6이 발생하는 경우 있음 해당 column 제거
        # input_df.dropna(subset=['TF_order'], inplace=True) # TF_order 에서 nan값 제거

        hero_col = [x for x in input_df.columns if x.startswith('Hero')]
        df_init = input_df[hero_col+features+target]
        # extract start point of each teamfight
        df_init = df_init.groupby(['MatchId', 'Map', 'Section', 'Team', 'Player', 'TF_order']).first()
        print('#nan in df_init', df_init.isna().sum())
        # pivot
        df_pivot = df_init.pivot_table(index=['MatchId', 'Map', 'Section', 'TF_order'], columns=['Team', 'Player'])
        print((np.unique(df_pivot.columns.get_level_values(0))))
        df_X = df_pivot.drop(columns=[x for x in df_pivot.columns.get_level_values(0) if x.startswith(target[0])])
        df_y = df_pivot[target].mean(axis=1)

        print('#nan in df_X: ', df_X.isna().sum().sum())
        print('#nan in df_y: ', df_y.isna().sum().sum())
        print('#nan in df_X: ', df_X.isna().sum())
        print('#nan in df_y: ', df_y.isna().sum())
        self.df_X = df_X 
        self.df_y = df_y

        X = df_X.values
        y = df_y.values
        print(f'X shape: {X.shape}')
        print(f'y shape: {y.shape}')

        self.X = X 
        self.y = y

        return X, y