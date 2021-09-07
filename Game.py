import pandas as pd
import math
import Codes
import os
import json
import itertools
import datetime

class Game:

    def __init__(self, moments_json_path='', pbp_file_path='', moments_processor=None, output_path=None,
                 pbp_processor=None, seconds_threshold=3, processed_data_path=None, using_cluster=False):
        """
        Game object constructor, used for processing pbp data and moments data of the same game
        :param moments_json_path: path to moments json
        :param pbp_file_path: path to play by play csv
        :param moments_processor: moments preprocessor
        :param pbp_processor: pbp preprocessor
        :param seconds_threshold: min possession length
        :param processed_data_path: path to processed data if exists, if None the game will be created from raw data
        """
        self.using_cluster = using_cluster
        self.output_path = output_path
        a = datetime.datetime.now()
        if processed_data_path is not None:
            self.load_game_from_jsons(processed_data_path)
        else:
            # Setting the seconds threshold used to determine the minimum length of a possession
            self.seconds_threshold = seconds_threshold

            # Preprocessing the moments data file (comes as a json inside a 7z)
            raw_moments_df = moments_processor.read_moments_file(moments_json_path)

            b = datetime.datetime.now()
            print(f'reading moments files time = {b - a}')

            self.game_id, self.game_date, self.home_team, self.home_players, self.visitor_team, self.visitor_players = \
                moments_processor.extract_game_meta_data(raw_moments_df)

            a = datetime.datetime.now()
            print(f'extracting game {self.game_id} meta data time = {a - b}')

            if self.if_preprocessed():
                os.remove(moments_processor.json_path)
                self.skip = True
                return
            else:
                self.skip = False

            self.moments_frame = moments_processor.preprocessing(raw_moments_df)

            # Preprocessing the pbp csv

            b = datetime.datetime.now()
            print(f'moments preprocessing time = {b - a}')

            self.pbp_data = pbp_processor.preprocess(pbp_file_path)
            a = datetime.datetime.now()
            print(f'preprocessing pbp time = {a - b}')

            # In some possessions there is no ball location
            for i, possession in self.moments_frame.iterrows():
                for moment in possession.moments:
                    if len(moment[5]) < 10:
                        print('Possession with less than 10 locations was found')

            if self.moments_frame is not None:
                # Merging the two datasets to create merged possessions
                self.merged_data, self.match_poss_index = \
                    self.merge_possessions_indexes(self.moments_frame, self.pbp_data, fuzzy=5)

            # Post merge moments preprocess:
            self.merged_data = moments_processor.separate_ids_from_coordinates(self.merged_data)
            self.merged_data = moments_processor.add_velocity_and_angels(self.merged_data)
            self.merged_data = moments_processor.detect_player_in_possession(self.merged_data)
            self.merged_data = moments_processor.detect_shots(self.merged_data)
            self.merged_data = moments_processor.detect_passes(self.merged_data)

            os.remove(moments_processor.json_path)

    def if_preprocessed(self):
        processed_games = os.listdir(self.output_path)
        if str(self.game_id) in processed_games:
            return True
        return False

    def merge_possessions_indexes(self, m_poss, pbp_poss, fuzzy=5, get_possessions_pairings=False):
        merged_data = list()
        match_poss_index = list()
        pbp_poss['matched'] = 0
        m_poss['matched'] = 0

        for curr_fuzzy in range(1, fuzzy):

            for quarter in range(1, 4):
                mom_by_quarter, pbp_by_quarter = self.filter_frame_by_quarter(m_poss, pbp_poss, quarter)
                for i, moments_possession in mom_by_quarter.iterrows():

                    if moments_possession['matched'] == 0:

                        for j in range(len(pbp_by_quarter)):
                            pbp_possession = pbp_by_quarter.iloc[j]

                            if pbp_possession['matched'] == 0:
                                if self.check_possessions_fit(pbp_possession, moments_possession, curr_fuzzy):
                                    # Mark the possession in both data sets as matched
                                    pbp_poss.at[j, 'matched'] = 1
                                    m_poss.at[i, 'matched'] = 1

                                    merged_data.append(self.create_possession_from_data(moments_possession, pbp_possession))

                                    if get_possessions_pairings:
                                        fit_pos_times_dict = self.create_matched_poss_dict(moments_possession,
                                                                                           pbp_possession)
                                        match_poss_index.append(fit_pos_times_dict)
                                    break

        merged_data_df = pd.DataFrame(merged_data)
        merged_data_df = merged_data_df.reindex()
        merged_data_df.index.name = 'pos_index'
        merged_data_df = merged_data_df.reset_index(drop=True)
        return merged_data_df, match_poss_index

    @staticmethod
    def create_matched_poss_dict(moments_possession, pbp_possession):
        fit_pos_times_dict = {
            'quarter': pbp_possession['quarter'],
            'pbp_start': pbp_possession['start_time'],
            'pbp_end': pbp_possession['end_time'],
            'mom_start': moments_possession['start_time'],
            'mom_end': moments_possession['end_time']
        }
        return fit_pos_times_dict

    @staticmethod
    def filter_frame_by_quarter(m_poss, pbp_poss, quarter):
        pbp_by_quarter = pbp_poss[pbp_poss['quarter'] == quarter]
        mom_by_quarter = m_poss[m_poss['quarter'] == quarter]
        return mom_by_quarter, pbp_by_quarter

    @staticmethod
    def check_possessions_fit(pbp_possession, moments_possession, fuzzy):
        if (pbp_possession['start_time'] < moments_possession['start_time'] + fuzzy):
            if (pbp_possession['end_time'] > moments_possession['end_time'] - fuzzy):
                if pbp_possession['duration'] + fuzzy > moments_possession['duration']:
                    return True
        return False


    def create_possession_from_data(self, moments_data, pbp_data):

        pbp_data['moments'] = self.format_moments(moments_data['moments'])
        pbp_data['home_players'], pbp_data['visitor_players'] = \
            self.retrieve_possession_players_list(pbp_data['moments'])
        pbp_data['moments_start_time'] = moments_data['start_time']
        pbp_data['moments_end_time'] = moments_data['end_time']
        pbp_data['moments_duration'] = moments_data['duration']
        pbp_data['home_team_id'] = Codes.TEAM_IDS[self.home_team]
        pbp_data['visitor_team_id'] = Codes.TEAM_IDS[self.visitor_team]
        return pbp_data

    def format_moments(self, moments):
        moments_frame = pd.DataFrame(moments,
                             columns=['event_id', 'moment_id', 'quarter', 'quarter_clock', 'shot_clock', 'locations'])
        # moments_frame = moments_frame.drop(columns=['event_id', 'quarter'])
        moments_frame = moments_frame.drop(columns=['quarter'])
        return moments_frame

    def retrieve_possession_players_list(self, moments):
        home_players = list()
        visitor_players = list()
        moment = moments.iloc[0]
        for location in moment['locations']:
            team_id = location[0]
            player_id = location[1]
            if player_id == -1:
                pass
            elif Codes.TEAM_CODES[team_id] == self.home_team:
                home_players.append(player_id)
            else:
                visitor_players.append(player_id)
        if len(home_players) != 5:
            print('hi')
        if len(visitor_players) != 5:
            print('hi')
        return home_players, visitor_players

    @staticmethod
    def convert_players_string_to_list(players_string):
        players_string = players_string.replace('[', '')
        players_string = players_string.replace(']', '')
        players_string = players_string.replace(',', '')
        players_list = players_string.split()
        return list(map(int, players_list))


    def save_merged_data(self, path):
        # Save each locations data as sepeatated frame:
        locations_path = os.path.join(path, str(self.game_id))
        if not os.path.exists(locations_path):
            os.makedirs(locations_path)
        for i, row in self.merged_data.iterrows():
            row['moments'].to_json(os.path.join(locations_path, str(i) + '.json'))
        df_to_write = self.merged_data.drop(columns='moments')
        df_to_write.to_csv(os.path.join(locations_path, 'game_file_' + str(self.game_id) + '.csv'))

    def save_game_as_jsons(self, path):
        if not self.skip:
            try:
                dir_path = os.path.join(path, str(self.game_id))
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                self.save_game_meta_data(dir_path)
                data_path = os.path.join(dir_path, 'merged_data.json')
                self.merged_data = self.convert_locations(self.merged_data)
                self.merged_data.to_json(data_path)
            except:
                print(f'Error saving game {self.game_id} as jsons ')
        else:
            print(f'Game {self.game_id} already preprocessed')

    def save_game_meta_data(self, path):
        meta_data = {
            'game_id': str(self.game_id),
            'game_date': self.game_date,
            'home_team': self.home_team,
            'home_players': self.home_players,
            'visitor_team': self.visitor_team,
            'visitor_players': self.visitor_players,
        }
        json_path = os.path.join(path, 'meta_data.json')
        with open(json_path, 'w') as fp:
            json.dump(meta_data, fp)

    def load_game_from_jsons(self, path):
        merged_data_path = os.path.join(path, 'merged_data.json')
        if os.path.exists(merged_data_path):
            meta_data = self.load_game_meta_data_from_json(path)
            self.assign_meta_data(meta_data)
            self.merged_data = pd.read_json(merged_data_path)
            # While writing the df sometimes the format is changed to dict therefore this method is crucial
            if not isinstance(self.merged_data.moments.iloc[0]['locations'], pd.DataFrame):
                self.convert_locations_to_df(self.merged_data)

    @staticmethod
    def convert_locations(merged_data):
        for i, possession in merged_data.iterrows():
            merged_data.at[i, 'moments']['locations'] = possession.moments['locations'].values
        return merged_data

    @staticmethod
    def convert_locations_to_df(merged_data):
        locations_df_columns = Codes.get_locations_df_column_names()
        for i, possession in merged_data.iterrows():
            possession.moments['locations'] = pd.DataFrame(possession.moments['locations'], columns=locations_df_columns)


    def assign_meta_data(self, meta_data):
        self.game_id = meta_data['game_id']
        self.game_date = meta_data['game_date']
        self.home_team = meta_data['home_team']
        self.home_players = meta_data['home_players']
        self.visitor_team = meta_data['visitor_team']
        self.visitor_players = meta_data['visitor_players']

    def load_game_meta_data_from_json(self, path):
        json_path = os.path.join(path, 'meta_data.json')
        with open(json_path, 'r') as json_file:
            meta_data = json.load(json_file)
        return meta_data

    def calculate_distance_from_ball(self, ball_x, ball_y, player_x, player_y):
        x_dis = math.pow(player_x - ball_x, 2)
        y_dis = math.pow(player_y - ball_y, 2)
        return math.sqrt(x_dis + y_dis)

    def convert_data_to_LSTM_model_inputs(self, only_unsuccessful=False, separated_def_inputs=False):

        X = list()
        Y = list()
        for i, row in self.merged_data.iterrows():
            if only_unsuccessful:
                if row['miss'] != 1 and row['turnover'] != 1 and row['violation'] != 1:
                    continue

            pos_data = dict()
            is_home_attacking = self.home_team == row['attacking_team']

            ball, attacking, defending = \
                self.convert_locations_to_keras_input(row['moments'], is_home_attacking, separated_def_inputs)
            pos_data['ball_locations'] = ball
            pos_data['attacking_locations'] = attacking
            pos_data['defending_locations'] = defending
            Y.append(defending)

            players_ids = self.convert_players_to_keras_input(row['home_players'], row['visitor_players'], is_home_attacking)
            pos_data = {**pos_data, **players_ids}

            pos_data['attacking_team_id'], pos_data['defending_team_id'] = \
                self.convert_team_ids_to_keras_input(is_home_attacking)

            X.append(pos_data)

        X = pd.DataFrame(X)
        Y = pd.Series(Y)
        return X, Y

    def check_if_possession_successful(self, series):

        if series['2pt'] == 1 or series['3pt'] == 1 or series['foul'] == 1 or series['shooting_foul'] == 1:
            return True
        else:
            return False

    def convert_locations_to_keras_input(self, moments_dict, is_home_attacking, separated_def_inputs=False):
        d = moments_dict['locations']
        ball = list()
        attacking = list()
        defending = list()
        for key, value in d.items():
            moment_locs = list()
            for i in range(len(value)):
                x = value[i][2]
                y = value[i][3]
                z = value[i][4]
                loc = [x, y, z]
                moment_locs.append(loc)
            if len(moment_locs) != 11:
                continue
            ball_location = moment_locs[0]
            if is_home_attacking:
                attacking_team_locs = moment_locs[1:6]
                defending_team_locs = moment_locs[6:]
            else:
                attacking_team_locs = moment_locs[6:]
                defending_team_locs = moment_locs[1:6]

            if len(defending_team_locs) != 5:
                defending_team_locs = defending_team_locs[:5]
            if len(attacking_team_locs) != 5:
                attacking_team_locs = attacking_team_locs[:5]

            # for i in range(len(defending))

            attacking_team_locs = list(itertools.chain.from_iterable(attacking_team_locs))
            if not separated_def_inputs:
                defending_team_locs = list(itertools.chain.from_iterable(defending_team_locs))
            ball.append(ball_location)
            attacking.append(attacking_team_locs)
            defending.append(defending_team_locs)
        return ball, attacking, defending

    def convert_players_to_keras_input(self, home_players, visitor_players, is_home_attacking):
        players = dict()

        for i, player in enumerate(home_players):
            if not is_home_attacking:
                i += 5
            players['player_' + str(i)] = Codes.get_player_embed_id(player)

        for i, player in enumerate(visitor_players):
            if is_home_attacking:
                i += 5
            players['player_' + str(i)] = Codes.get_player_embed_id(player)

        return players

    def convert_team_ids_to_keras_input(self, is_home_attacking):
        if is_home_attacking:
            return Codes.Embed_Team_IDs[self.home_team], Codes.Embed_Team_IDs[self.visitor_team]
        else:
            return Codes.Embed_Team_IDs[self.visitor_team], Codes.Embed_Team_IDs[self.home_team]