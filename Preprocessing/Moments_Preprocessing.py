import pandas as pd
import numpy as np
import Codes
import py7zr
import math
import os

class Moments_Processor():

    def __init__(self, using_cluster=False):
        self.using_cluster = using_cluster

    def preprocessing(self, moments_frame):
        # Process moments
        moments_frame = self.events_to_moments(moments_frame['events'])
        moments_frame = self.remove_damaged_moments(moments_frame)
        possessions_frame = self.separate_moments_by_shot_clock(moments_frame)
        possessions_frame = self.remove_24_possessions(possessions_frame)
        return possessions_frame

    def read_moments_file(self, file_path):
        file_path = self.check_file_format(file_path)
        self.json_path = file_path
        df = pd.read_json(file_path)
        return df

    def extract_game_meta_data(self, moments_frame):
        game_id = str(moments_frame['gameid'].iloc[0])
        game_date = moments_frame['gamedate'].iloc[0]
        first_event = moments_frame.iloc[0]['events']
        home_team, visitor_team_team = self.extract_teams_names(first_event)
        home_players, visitor_players = self.extract_players(first_event)
        return game_id, game_date, home_team, home_players, visitor_team_team, visitor_players

    def events_to_moments(self, events):
        """
        Each game is composed of multiple events that contain moments. Those events has multiple possessions in them
        and sometime cut a possession in the middle. This function merges all the events and remove duplicate moments
        in order to prepare the data to further processing.
        :param events: Events
        :return:
        """
        rows = list()
        for event in events.values:
            for moment in event['moments']:
                moment_dict = self.create_moment_dict(moment, int(event['eventId']))
                rows.append(moment_dict)
        df = pd.DataFrame(rows)
        df = self.remove_duplicate_moments(df)

        # Moments without shot clock usually has just one location (the ball's)
        df = self.remove_moments_without_shot_clock(df)

        # Sorting by event id and then quarter clock eliminates the possession merge phenomena
        df = df.astype({'quarter_clock': float, 'shot_clock': float})

        df = df.sort_values(by=['quarter_clock'], ascending=False)
        df = df.sort_values(by=['quarter', 'event_id'])

        df = df.reset_index(drop=True)
        return df

    def remove_damaged_moments(self, moments_frame):
        """
        Removes moments that lacks some of the locations for the players or the ball
        :param moments_frame: pd DataFrame with the moments data
        :return: pd DataFrame without the moments that lacks locations
        """
        for i, moment in moments_frame.iterrows():
            if len(moment['locations']) < 11:
                moments_frame.drop(i, inplace=True)
        moments_frame = moments_frame.reset_index(drop=True)
        return moments_frame

    def remove_24_possessions(self, possessions_frame):

        for i, possession in possessions_frame.iterrows():
            to_remove = True
            for moment in possession.moments:
                if moment[4] != 24:
                    to_remove = False
                    break

            if to_remove:
                possessions_frame.drop(i, inplace=True)

        return possessions_frame.reset_index(drop=True)

    def create_moment_dict(self, moment, event_id):
        moment = {'event_id': event_id,
                  'moment_id': moment[1],
                  'quarter': moment[0],
                  'quarter_clock': moment[2],
                  'shot_clock': moment[3],
                  'locations': moment[5]}
        return moment

    @staticmethod
    def remove_moments_without_shot_clock(df):
        df = df[~df.shot_clock.isnull()]
        return df

    @staticmethod
    def remove_duplicate_moments(df):
        # Must create inverted quarter field to sort by ascending quarter and descending quarter clock
        df['inverted_quarter'] = [4 - quarter + 1 for quarter in df['quarter']]
        df.sort_values(by=['inverted_quarter', 'quarter_clock'], inplace=True, ascending=False)
        df = df.drop('inverted_quarter', axis=1)
        df = df.drop_duplicates(subset=['quarter', 'quarter_clock'])
        return df

    def separate_moments_by_shot_clock(self, moments_frame):
        """
        Create possible separation to possessions using the shot clock in the moments data.
        The possessions will be compared to the possessions created by the play by play data
        :param moments_frame: dataframe that contains the moments data
        :return: dataframe with separated possessions
        """
        clock = 25
        possessions = list()
        moments = list()
        start_time = moments_frame.iloc[0]['quarter_clock']

        shot_clock_index = moments_frame.columns.get_loc('shot_clock')
        quarter_clock_index = moments_frame.columns.get_loc('quarter_clock')
        quarter_index = moments_frame.columns.get_loc('quarter')

        if len(moments_frame[moments_frame.shot_clock.notnull()]) == 0:
            return None

        for i, moment in enumerate(moments_frame.values):
            if math.isnan(moment[shot_clock_index]):
                moments.append(moment)
                clock = moment[shot_clock_index]

            elif moment[shot_clock_index] > clock or \
                    (clock == 24 and clock > moment[shot_clock_index] and len(moments) > 1):
                # New possession started
                # Create new possession
                # end_time = moment[quarter_clock_index]
                end_time = moments_frame.values[i - 1][quarter_clock_index] # Preventing quarter clock time skips
                new_possession_dict = {
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': start_time - end_time,
                    'quarter': moment[quarter_index],
                    'moments': moments
                }
                # Including only possessions that are longer than the seconds threshold seconds
                if start_time - end_time > Codes.POSSESSION_MIN_LENGTH:
                    # possessions[id] = new_possession
                    possessions.append(new_possession_dict)

                moments = list()
                moments.append(moment)
                start_time = moment[quarter_clock_index]
                clock = moment[shot_clock_index]
            else:
                moments.append(moment)
                clock = moment[shot_clock_index]

        possessions_df = pd.DataFrame.from_dict(possessions)
        return possessions_df

    def check_file_format(self, file_path):
        if self.using_cluster:
            slash = '/'
        else:
            slash = '\\'

        try:
            if file_path.endswith('7z'):
                with py7zr.SevenZipFile(file_path, 'r') as archive:
                    allfiles = archive.getnames()
                    if allfiles:
                        json_name = allfiles[0]
                        dir_path = file_path[:file_path.rfind(slash)]
                        if not os.path.exists(dir_path + slash + json_name[:-5] + '.csv'):
                            archive.extractall(dir_path)
                            file_path = dir_path + slash + json_name
                        else:
                            archive.close()
                            pass
        except:
            if archive is not None:
                archive.close()

        return file_path

    @staticmethod
    def extract_teams_names(event):
        home_team = event['home']['name'].upper()
        visitor_team = event['visitor']['name'].upper()
        return home_team, visitor_team

    @staticmethod
    def extract_players(event):
        home_players = event['home']
        visitor_players = event['visitor']
        return home_players, visitor_players

    @staticmethod
    def shot_taken(possession):
        if possession.make == 1 or possession.miss == 1:
            return True
        else:
            return False

    def detect_shots(self, possessions_df):

        for i, possession in possessions_df.iterrows():
            threshold = Codes.SHOT_HEIGHT_THRESHOLD
            if possession['block'] == 1:
                threshold = 7.5

            if not self.shot_taken(possession):
                shot_occurred = np.zeros(len(possession.moments['locations']))
                possession.moments['shots'] = shot_occurred
                continue

            last_shot_index = -1
            shot_status = 0

            # for i, row in enumerate(possession.moments['locations'].values()):
            for j, row in possession.moments['locations'].iterrows():
                ball_z = row['ball_z']
                if shot_status == 0:
                    if ball_z > threshold:
                        shot_status = 1
                        last_shot_index = int(j)
                else:
                    if ball_z < threshold:
                        shot_status = 0

            shot_occurred = np.zeros(len(possession.moments['locations']))

            if last_shot_index == -1:
                last_shot_index = len(possession.moments['player_in_possession']) - 1

            player_in_pos = possession.moments['player_in_possession']

            if possession.moments['player_in_possession'][last_shot_index] == -1:
                for j in range(last_shot_index, 0, -1):
                    if player_in_pos[j] != -1:
                        last_shot_index = j
                        break
            else:
                possessions_df.at[i, 'moments']['player_in_possession'] = \
                    player_in_pos[:last_shot_index + 1] + [-1] * (len(player_in_pos) - last_shot_index - 1)


            if last_shot_index != -1:
                shot_occurred[last_shot_index] = 1
            else:
                possessions_df.at[i, 'moments']['shots'] = shot_occurred
                print(f'No shot was detected, check possession {i}')  # No shot detected
                # Animations.animate_possession(possession, i=i)


            possessions_df.at[i, 'moments']['shots'] = shot_occurred

        return possessions_df

    def add_velocity_and_angels(self, possessions_df):
        for i, possession in possessions_df.iterrows():
            movement_vectors = []
            for j, curr_coordinates in enumerate(possession.moments['locations'].values):
                if j == 0:
                    vectors = self.calculate_vectors(curr_coordinates, curr_coordinates)
                else:
                    vectors = self.calculate_vectors(previous_coordinates, curr_coordinates)

                movement_vectors.append(vectors)
                previous_coordinates = curr_coordinates

            possessions_df.at[i, 'moments']['movement_vectors'] = pd.DataFrame(movement_vectors)

        return possessions_df

    @staticmethod
    def calculate_vectors(first_coordinates, second_coordinates):
        return (np.array(first_coordinates) - np.array(second_coordinates)) / 0.04

    @staticmethod
    def unit_vector(vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2':: """
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


    def separate_ids_from_coordinates(self, possessions_df):
        locations_df_column_names = Codes.get_locations_df_column_names()
        is_first_team_attacking = list()

        for i, possession in possessions_df.iterrows():

            if not isinstance(possessions_df.at[i, 'moments'], dict):
                possessions_df.at[i, 'moments'] = possession.moments.to_dict()
                possession.moments = possessions_df.at[i, 'moments']

            locations_np_arr = list()
            locations = list(possession.moments['locations'].values())
            is_first_team_attacking.append(self.detect_attacking_team_locations_order(possession))

            players_ids = set()

            for coordinates in locations:
                for player_coor in coordinates:
                    player_id = player_coor[1]

                    if player_id != -1 and len(players_ids) == 10 and player_id not in players_ids:
                        pass
                    elif player_id != -1 and player_id not in players_ids:
                        players_ids.add(player_id)

                    del player_coor[:2]
                    if player_id != -1:
                        del player_coor[-1]

                locations_np_arr.append(np.hstack(coordinates))
            locations_df = pd.DataFrame(locations_np_arr)
            locations_df.columns = locations_df_column_names
            possessions_df.at[i, 'moments']['locations'] = locations_df

        possessions_df['is_first_team_attacking'] = is_first_team_attacking
        return possessions_df

    @staticmethod
    def detect_attacking_team_locations_order(possession):
        attacking_team_id = int(Codes.TEAM_IDS[possession.attacking_team])
        first_locations = list(possession.moments['locations'].values())[0]
        if first_locations[1][0] == attacking_team_id:
            return 1
        else:
            return 0


    def detect_player_in_possession(self, possessions_df):
        for i, possession in possessions_df.iterrows():

            locations = possession.moments['locations']
            ball_location, attacking_players_locations = \
                self.get_attacking_or_defensive_team_locations(locations, possession.is_first_team_attacking)

            player_in_possession_ids = list()

            for j, attacking_locations in attacking_players_locations.iterrows():
                close_players = {}
                attacking_locations = self.convert_to_locations_tuples(attacking_locations)

                if isinstance(j, str):
                    j = int(j)

                ball_x, ball_y = ball_location.iloc[j][0], ball_location.iloc[j][1]

                for player_num, location in enumerate(attacking_locations):
                    player_distance_from_ball = self.calculate_distance_between_two_locations(ball_x, ball_y, location[0], location[1])
                    if player_distance_from_ball <= Codes.POSSESSION_THRESHOLD_FEET:
                        close_players[player_num] = player_distance_from_ball

                if len(player_in_possession_ids) > 0:
                    previous_player = player_in_possession_ids[-1]
                else:
                    previous_player = -1
                player_id = self.select_player_in_possession(close_players, previous_player)
                player_in_possession_ids.append(player_id)
            possessions_df.at[i, 'moments']['player_in_possession'] = player_in_possession_ids

        return possessions_df

    @staticmethod
    def get_attacking_or_defensive_team_locations(locations, is_first_team_attacking):
        # Return the offensive or defensive team locations
        ball_locations = locations.iloc[:, 0:3]
        if is_first_team_attacking:
            attacking_team_locations = locations.loc[:, 'player_0_x':'player_4_z']
        else:
            attacking_team_locations = locations.loc[:, 'player_5_x':'player_9_z']
        return ball_locations, attacking_team_locations

    @staticmethod
    def convert_to_locations_tuples(locations):
        locations_tuples = list()
        for i in range(0, len(locations), 2):
            locations_tuples.append(locations.iloc[i:i+2])
        return locations_tuples

    @staticmethod
    def calculate_distance_between_two_locations(ball_x, ball_y, player_x, player_y):
        x_dis = math.pow(player_x - ball_x, 2)
        y_dis = math.pow(player_y - ball_y, 2)
        return math.sqrt(x_dis + y_dis)

    @staticmethod
    def select_player_in_possession(players_distance_dict, previous_player):
        # If the previous player who was in control of the ball is still in a relatively close distance from it
        # he is probably still the player who controls it
        if previous_player != -1 and previous_player in players_distance_dict.keys():
            return previous_player

        player_in_possession = -1 # Number of the player in the home / away team IDs
        min_distance =  float('inf')

        for key, value in players_distance_dict.items():
            if value < min_distance:
                min_distance = value
                player_in_possession = key
        return player_in_possession

    def detect_passes(self, possessions_df):
        passes_list = list()

        for i, possession in possessions_df.iterrows():
            bad_pass_flag = False
            if 'bad pass' in possession.description:
                bad_pass_flag = True

            locations = possession.moments['locations']
            players_in_possession = possession.moments['player_in_possession']
            possible_passer = players_in_possession[0]
            ongoing_pass = False
            passes = dict()

            for j, ball_handler in enumerate(players_in_possession):
                if possession.moments['shots'][j] == 1:
                    break

                # Pass start
                if ball_handler == -1 and not ongoing_pass:
                    ongoing_pass = True
                    pass_start_location = locations.iloc[j].iloc[0:3] # Ball location in the beginning of the pass
                    pass_start_index = j

                elif ball_handler != possible_passer and not ongoing_pass: # Possible handoff
                    passes[j] = {'players': (possible_passer, ball_handler),
                                 'type': 'handoff',
                                 'locations': (locations.iloc[j-1].iloc[0:3], locations.iloc[j].iloc[0:3]),
                                 'index': j}
                    possible_passer = ball_handler # Fix the handoffs that continue for more than 1 moment

                # Pass end
                elif ball_handler != -1 and possible_passer != ball_handler:
                    if not pass_start_index == 0:
                        ongoing_pass = False
                        pass_end_location = locations.iloc[j].iloc[0:3]  # Ball location in the beginning of the pass
                        passes[j] = {'players': (possible_passer, ball_handler),
                                     'type': 'pass',
                                     'locations': (pass_start_location, pass_end_location),
                                     'index': (pass_start_index - 1, j)}
                    possible_passer = ball_handler

                elif ball_handler == possible_passer and ongoing_pass:
                    ongoing_pass = False

            # Adding relevant passes
            # If the possession ended with a bad pass add the last pass that was detected:
            if ongoing_pass and bad_pass_flag:
                j, target_player = self.get_target_player(j, locations, possession, possible_passer)

                pass_end_location = locations.iloc[j].iloc[0:3]  # Ball location in the beginning of the pass
                passes[j] = {'players': (possible_passer, target_player),
                             'type': 'bad pass',
                             'locations': (pass_start_location, pass_end_location),
                             'index': (pass_start_index, j - 1)}

            pass_index = np.zeros(len(players_in_possession))
            for key, pass_details in passes.items():
                if pass_details['players'][0] == pass_details['players'][1]:
                    continue # Probably not a pass
                if pass_details['type'] == 'pass' or pass_details['type'] == 'bad pass':
                    start = pass_details['index'][0]
                    finish = pass_details['index'][1]
                    pass_index[start:finish] = 1
                else:
                    pass_index[pass_details['index']] = 2

            possessions_df.at[i, 'moments']['pass_index'] = pass_index
            passes_list.append(passes)

        possessions_df['passes'] = passes_list
        return possessions_df

    def get_target_player(self, j, locations, possession, possible_passer):
        ball_location, attacking_locations = \
            self.get_attacking_or_defensive_team_locations(locations, possession.is_first_team_attacking)
        ball_location = ball_location.iloc[-1]
        attacking_locations = attacking_locations.iloc[-1]
        players_dis_from_ball = {}
        attacking_locations = self.convert_to_locations_tuples(attacking_locations)
        if isinstance(j, str):
            j = int(j)
        ball_x, ball_y = ball_location.iloc[0], ball_location.iloc[1]
        for player_num, location in enumerate(attacking_locations):
            if player_num != possible_passer:
                player_distance_from_ball = self.calculate_distance_between_two_locations(ball_x, ball_y,
                                                                                          location[0], location[1])
                players_dis_from_ball[player_num] = player_distance_from_ball
        target_player = self.select_player_in_possession(players_dis_from_ball, possible_passer)
        return j, target_player

    @staticmethod
    def list_of_zeroes(n):
        return [0] * n

    def calculate_possible_passes(self, possession):
        # Calculation from: https://squared2020.com/2017/09/21/building-a-simple-spatial-analytic-passing-lane-coverage/
        locations = possession.moments['locations']
        ball_location, offense_players_locations = \
            self.get_attacking_or_defensive_team_locations(locations, possession.is_first_team_attacking)
        ball_location, defense_players_locations = \
            self.get_attacking_or_defensive_team_locations(locations, not possession.is_first_team_attacking)


        offense_players_locations = offense_players_locations.iloc[50]
        offense_players_locations = self.convert_to_locations_tuples(offense_players_locations)
        defense_players_locations = defense_players_locations.iloc[50]
        defense_players_locations = self.convert_to_locations_tuples(defense_players_locations)

        ball_handler_id = possession.moments['player_in_possession'][50]
        handler_locations = defense_players_locations[ball_handler_id]
        possible_passes = list()
        for i, attacker_location in enumerate(offense_players_locations):
            if i != ball_handler_id:  # Skip the handler
                P = handler_locations.values - attacker_location.values
                Ds = handler_locations.values - np.vstack(defense_players_locations)

                P_size = np.sqrt(np.dot(P, P.T))
                Ds_size = sum(np.sqrt(np.dot(Ds, Ds.T)))

                distances = Ds_size * np.sin(np.arccos(np.dot(P, Ds.T) / (P_size * Ds_size)))
                distances = np.nan_to_num(distances)

                if not np.any(distances <= Codes.PASSING_LANE_MIN_DIS):
                    possible_passes.append(i)

        return possible_passes
