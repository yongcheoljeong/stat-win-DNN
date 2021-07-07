import pandas as pd 
from random import *
from sklearn.preprocessing import MinMaxScaler

def PositionInfo(input_df):
    # dissect into x, y, z position
    input_df['Position'] = input_df['Position'].str.replace('(', '')
    input_df['Position'] = input_df['Position'].str.replace(')', '')
    input_df['Position_x'] = input_df['Position'].str.split(',').str[0].astype('float32')
    input_df['Position_y'] = input_df['Position'].str.split(',').str[2].astype('float32')
    input_df['Position_z'] = input_df['Position'].str.split(',').str[1].astype('float32')
    output_df = input_df

    return output_df

def SkillUsedTimestampInfo(input_df):
    # dissect into 0, 1 binary info
    input_df.loc[input_df['Cooldown1%/s']>0, 'Cooldown1%/s'] = 1
    input_df.loc[input_df['Cooldown1%/s']<0, 'Cooldown1%/s'] = 0
    input_df.loc[input_df['Cooldown2%/s']>0, 'Cooldown2%/s'] = 1
    input_df.loc[input_df['Cooldown2%/s']<0, 'Cooldown2%/s'] = 0
    input_df.loc[input_df['CooldownCrouching%/s']>0, 'CooldownCrouching%/s'] = 1
    input_df.loc[input_df['CooldownCrouching%/s']<0, 'CooldownCrouching%/s'] = 0
    input_df.loc[input_df['CooldownSecondaryFire%/s']>0, 'CooldownSecondaryFire%/s'] = 1
    input_df.loc[input_df['CooldownSecondaryFire%/s']<0, 'CooldownSecondaryFire%/s'] = 0
    input_df.loc[:,'UltimateCharge'] = input_df.loc[:,'UltimateCharge']/100
    output_df = input_df
    return output_df

def CategoricalEncoding(input_df):
    df_init = input_df.reset_index()
    # Blind Team/Player
    match_ids = df_init['MatchId'].unique()
    # match_id별로 team name 구하고 random하게 0, 1로 변환/Player는 팀별로 0, 1, 2 ,3, 4, 5로 변환
    for matchid in match_ids:
        map_names = df_init.loc[df_init['MatchId']==matchid, 'Map'].unique()
        for mapid in map_names:
            scaler = MinMaxScaler()
            df_init.loc[((df_init['MatchId']==matchid) & (df_init['Map']==mapid)), ['Position_x', 'Position_y', 'Position_z', 'RCP']] = scaler.fit_transform(df_init.loc[((df_init['MatchId']==matchid) & (df_init['Map']==mapid)), ['Position_x', 'Position_y', 'Position_z', 'RCP']])
            section_names = df_init.loc[((df_init['MatchId']==matchid) & (df_init['Map']==mapid)), 'Section'].unique()
            for sectionid in section_names:
                team_names = df_init.loc[((df_init['MatchId']==matchid) & (df_init['Map']==mapid) & (df_init['Section']==sectionid)), 'Team'].unique()
                # team_one = choice(team_names) 
                team_one = 'NYE' # scrimlog는 teamone이 항상 NYE
                team_two = [x for x in team_names if x != team_one][0]
                team_one_players = df_init.loc[((df_init['MatchId']==matchid) & (df_init['Map']==mapid) & (df_init['Section']==sectionid) & (df_init['Team']==team_one)), 'Player'].unique()
                team_two_players = df_init.loc[((df_init['MatchId']==matchid) & (df_init['Map']==mapid) & (df_init['Section']==sectionid) & (df_init['Team']==team_two)), 'Player'].unique()
                team_one_player_dict = {player:i for i, player in enumerate(team_one_players)}
                team_two_player_dict = {player:i for i, player in enumerate(team_two_players)}
                # replace
                df_init.loc[((df_init['MatchId']==matchid) & (df_init['Map']==mapid) & (df_init['Section']==sectionid)), 'Team'] = df_init.loc[((df_init['MatchId']==matchid) & (df_init['Map']==mapid) & (df_init['Section']==sectionid)), 'Team'].replace(team_one, 0)
                df_init.loc[((df_init['MatchId']==matchid) & (df_init['Map']==mapid) & (df_init['Section']==sectionid)), 'Team'] = df_init.loc[((df_init['MatchId']==matchid) & (df_init['Map']==mapid) & (df_init['Section']==sectionid)), 'Team'].replace(team_two, 1)

                df_init.loc[((df_init['MatchId']==matchid) & (df_init['Map']==mapid) & (df_init['Section']==sectionid)), 'TF_winner'] = df_init.loc[((df_init['MatchId']==matchid) & (df_init['Map']==mapid) & (df_init['Section']==sectionid)), 'TF_winner'].replace(team_one, 0)
                df_init.loc[((df_init['MatchId']==matchid) & (df_init['Map']==mapid) & (df_init['Section']==sectionid)), 'TF_winner'] = df_init.loc[((df_init['MatchId']==matchid) & (df_init['Map']==mapid) & (df_init['Section']==sectionid)), 'TF_winner'].replace(team_two, 1)
                df_init.loc[((df_init['MatchId']==matchid) & (df_init['Map']==mapid) & (df_init['Section']==sectionid)), 'TF_winner'] = df_init.loc[((df_init['MatchId']==matchid) & (df_init['Map']==mapid) & (df_init['Section']==sectionid)), 'TF_winner'].replace('draw', -1)
                # drop TF draw
                df_init = df_init[df_init['TF_winner'] != -1]

                df_init.loc[((df_init['MatchId']==matchid) & (df_init['Map']==mapid) & (df_init['Section']==sectionid)), 'Player'] = df_init.loc[((df_init['MatchId']==matchid) & (df_init['Map']==mapid) & (df_init['Section']==sectionid)), 'Player'].replace(team_one_player_dict)
                df_init.loc[((df_init['MatchId']==matchid) & (df_init['Map']==mapid) & (df_init['Section']==sectionid)), 'Player'] = df_init.loc[((df_init['MatchId']==matchid) & (df_init['Map']==mapid) & (df_init['Section']==sectionid)), 'Player'].replace(team_two_player_dict)

    # Hero One-Hot Encoding
    df_init = pd.get_dummies(data = df_init, columns = ['Hero'], prefix = 'Hero')

    output_df = df_init.set_index(['MatchId', 'Map', 'Section', 'Timestamp', 'Team', 'Player'])
    return output_df

def scale_data(input_df):
    pass 
