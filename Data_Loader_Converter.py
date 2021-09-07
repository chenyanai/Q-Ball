from tqdm import tqdm
from Game import Game
import pandas as pd
import numpy as np
import pickle
import Codes
import os

class Data_Loader:
    def __init__(self, path, episodes_save_path, sequence_length=25, load_games_num=20, skip_frames=0, transpose=True,
                 create_new_episodes_data=False, offense=True, load_pickle=True, pool=None):
        self.sequence_length = sequence_length
        self.skip_frames = skip_frames
        self.divide_locations = [90, 50, 20] + [90, 50] * 10
        self.max_velocity = 0
        self.load_games_num = load_games_num
        self.save_episodes = create_new_episodes_data
        self.transpose = transpose
        self.offense = offense
        self.next_episode_id = 0
        self.next_pickle = 0
        self.pool = pool
        self.skipped_episodes = 0

        self.episodes_save_path = episodes_save_path

        if create_new_episodes_data:
            self.create_episode_pickles(path)

        self.pickle_list = os.listdir(self.episodes_save_path)
        if load_pickle:
            self.load_next_pickle()

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    def load_pickles(self):
        episodes = list()
        for file in os.listdir(self.episodes_save_path):
            pickle_path = os.path.join(self.episodes_save_path, file)
            with open(pickle_path, 'rb') as handle:
                game_episodes = pickle.load(handle)
                episodes = episodes + game_episodes
        # self.episodes = episodes
        return episodes

    def create_episode_pickles(self, data_path):

        jobs = list()
        for folder in tqdm(os.listdir(data_path)):
            possible_pickle_path = os.path.join(self.episodes_save_path, folder + '_episodes.pickle')
            if not os.path.exists(possible_pickle_path): # Check if this game was already preprocessed
                job = self.pool.apply_async(self.game_to_episodes, args=(data_path, folder))
                jobs.append(job)

        for job in jobs:
            job.get()

        self.pool.close()
        self.pool.join()

    def game_to_episodes(self, data_path, folder):
        # Load game from json using the Game object constructor
        game = Game(processed_data_path=os.path.join(data_path, folder))
        episodes = self.create_episodes_from_game(game)
        # Save the episodes as pickle
        path = os.path.join(self.episodes_save_path, str(game.game_id) + '_episodes.pickle')
        with open(path, 'wb') as handle:
            pickle.dump(episodes, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def create_episodes_from_game(self, game):
        episodes = list()
        for i, possession in game.merged_data.iterrows():
            # Make sure the possession is long enough to be processed
            if len(possession.moments['locations']) > self.skip_frames * self.sequence_length:
                is_attacking_the_left = self.is_attacking_the_left_side(possession)
                episode = self.create_episode_from_possession(game, i, possession, is_attacking_the_left)
                episodes.append(episode)

                # Transpose locations and create another episode doubling the possession numbers
                if self.transpose:
                    self.transpose_locations(game, i)
                    is_attacking_the_left = int(not is_attacking_the_left)  # Change the attacking direction
                    episode = self.create_episode_from_possession(game, i, possession, is_attacking_the_left)
                    episodes.append(episode)
        return episodes

    def load_games_data(self, data_path):
        games = list()
        i = 0
        for folder in os.listdir(data_path):
            game = Game(processed_data_path=os.path.join(data_path, folder))
            games.append(game)
            i += 1
            if i == self.load_games_num:
                break
        return games

    def convert_games_to_DQN_input(self, games):
        episodes = list()
        for game in games:
            for i, possession in game.merged_data.iterrows():

                episode = self.create_episode_from_possession(game, i, possession)
                episodes.append(episode)
                if self.transpose:
                    # Transpose locations and create another episode doubling the possession numbers
                    self.transpose_locations(game, i)
                    episode = self.create_episode_from_possession(game, i, possession)
                    episodes.append(episode)

            if self.save_episodes:
                path = os.path.join(self.episodes_save_path, str(game.game_id) + '_episodes.pickle')
                with open(path, 'wb') as handle:
                    pickle.dump(episodes, handle, protocol=pickle.HIGHEST_PROTOCOL)
                episodes = list()

        self.episodes = episodes

    def create_episode_from_possession(self, game, i, possession, is_attacking_the_left):
        self.is_first_team_attacking = possession.is_first_team_attacking
        states, discrete_actions, continuous_actions, reward = self.possession_to_RL_input(possession, is_attacking_the_left)
        shot_taken = possession.make or possession.miss

        if self.skip_frames > 0:
            states, discrete_actions, continuous_actions = \
                self.adjust_data_to_frame_skipping(states, discrete_actions, shot_taken)

        game_meta_data = self.get_possession_meta_data(possession)
        episode_sequences = self.convert_episode_to_model_input(game_meta_data, states, discrete_actions,
                                                                continuous_actions, reward)
        episode = {
            'game_id': game.game_id,
            'possession_num': i,
            'sequences': episode_sequences
        }
        return episode

    def adjust_data_to_frame_skipping(self, states, discrete_actions, shot_taken):

        if shot_taken and 1 in discrete_actions.shot_index.values:
            start_index = discrete_actions[discrete_actions.shot_index == 1].index[0]
            if start_index == 0:
                discrete_actions.at[0, 'shot_index'] = 0
                discrete_actions.at[len(discrete_actions) - 1, 'shot_index'] = 1
                start_index = len(discrete_actions) - 1
        else:
            start_index = len(states) - 1

        next_save = 0
        saved_states_indexes = list()
        merged_discrete_actions = list()
        accumulated_discrete_actions = [0] * len(discrete_actions.columns)

        for i in range(start_index, 0, -1):
            if next_save == 0:
                current_discrete_actions = discrete_actions.iloc[i]
                discrete_actions_lists = [current_discrete_actions, accumulated_discrete_actions]
                current_discrete_actions = [sum(x) for x in zip(*discrete_actions_lists)]
                merged_discrete_actions.append(current_discrete_actions)
                saved_states_indexes.append(i)

                next_save = self.skip_frames - 1
            else:
                if 1 in discrete_actions:
                    action_id = discrete_actions.index(1)
                    accumulated_discrete_actions[action_id] = 1
                next_save -= 1

        merged_discrete_actions.reverse()
        merged_discrete_actions = pd.DataFrame(merged_discrete_actions)
        merged_discrete_actions.columns = discrete_actions.columns
        states = states.filter(sorted(saved_states_indexes), axis=0).reset_index(drop=True)

        continuous_actions = list()
        relevant_locations = self.get_relevant_team_locations(states, len(states.columns) - 23)
        for i, row in relevant_locations.iterrows():
            if i > 0:
                curr_continuous_actions = self.compute_new_velocity(row, previous_locations)
            else:
                curr_continuous_actions = self.compute_new_velocity(row, row)
            continuous_actions.append(curr_continuous_actions)
            previous_locations = row

        continuous_actions = pd.DataFrame(continuous_actions)
        return states, merged_discrete_actions, continuous_actions

    def convert_episode_to_model_input(self, meta_data, states, discrete_actions, continuous_actions, reward):

        episode_sequences = list()
        curr_reward = 0
        done = False
        for i, row in states.iterrows():
            if i >= self.sequence_length:
                if self.offense:
                    # Each dictionary contains the state and actions of a specific time
                    states_seq = states[i - self.sequence_length : i]
                    discrete_actions_seq = discrete_actions.iloc[i]
                    continuous_actions_seq = continuous_actions.iloc[i]
                else:
                    # Each dictionary contains the state and actions of a specific time
                    states_seq = states[i - self.sequence_length: i]
                    discrete_actions_seq = discrete_actions[i - self.sequence_length: i]
                    states_seq = pd.concat([states_seq, discrete_actions_seq], axis=1)
                    discrete_actions_seq = discrete_actions.iloc[i]
                    continuous_actions_seq = continuous_actions.iloc[i]

                # Finish the episode when the shot is taken or when the possession ends
                if i == len(states) - 1 or discrete_actions.iloc[i]['shot_index'] == 1:
                    curr_reward = reward
                    done = True

                episode_sequence = {
                    'state': (meta_data, states_seq),
                    'action': (discrete_actions_seq, continuous_actions_seq),
                    'reward': curr_reward,
                    'done': done
                }
                episode_sequences.append(episode_sequence)

                if done:
                    break

        return episode_sequences

    def compute_new_velocity(self, curr_locations, previous_locations):
        velocity = (curr_locations - previous_locations) / (self.sequence_length * 0.04)
        return velocity / self.max_velocity

    def get_episode_meta_data(self):
        if self.next_episode_id < len(self.episodes):
            episode = self.episodes[self.next_episode_id]
            return episode['game_id'], episode['possession_num']
        else:
            return None

    def get_possession_meta_data(self, possession):

        if possession.is_first_team_attacking:
            attacking_players = possession.home_players
            attacking_team_id = Codes.Embed_Team_IDs[Codes.TEAM_CODES[possession.home_team_id]]
            defending_players = possession.visitor_players
            defending_team_id = Codes.Embed_Team_IDs[Codes.TEAM_CODES[possession.visitor_team_id]]
            attacking_home_court = 1
        else:
            attacking_players = possession.visitor_players
            attacking_team_id = Codes.Embed_Team_IDs[Codes.TEAM_CODES[possession.visitor_team_id]]
            defending_players = possession.home_players
            defending_team_id = Codes.Embed_Team_IDs[Codes.TEAM_CODES[possession.home_team_id]]
            attacking_home_court = 0

        meta_data = {
            'attacking_players': [Codes.get_player_embed_id(player_id) for player_id in attacking_players],
            'attacking_team_id': attacking_team_id,
            'defending_players': [Codes.get_player_embed_id(player_id) for player_id in defending_players],
            'defending_team_id': defending_team_id,
            'meta_data': [attacking_home_court, possession.quarter]
        }

        meta_data['attacking_players_stats'] = self.create_players_stats_vector(attacking_players)
        meta_data['defending_players_stats'] = self.create_players_stats_vector(defending_players, defending=True)

        return meta_data

    def create_players_stats_vector(self, players, defending=False):
        players_stats = list()
        for player in players:
            stats = Codes.get_player_stats_by_id(player, defending=defending)
            players_stats += stats
        return players_stats

    def possession_to_RL_input(self, possession, is_attacking_the_left):
        reward = self.compute_reward(possession)
        states, discrete_actions, continuous_actions = \
            self.convert_moments_to_actions_and_states(possession.moments, possession.passes, is_attacking_the_left)

        return states, discrete_actions, continuous_actions, reward

    def compute_reward(self, possession):
        reward = 0
        shoots_num = 2
        if possession['2pt'] and possession['make']:
            reward = 2
        elif possession['3pt'] and possession['make']:
            reward = 3
        elif possession['offensive_rebound']:
            reward = 0.5
        elif possession['shooting_foul'] and not possession['And1']:
            if possession['3pt']:
                shoots_num = 3
            reward = self.compute_player_ft_reward(possession.player_name, shoots_num=shoots_num)
        elif possession['bad pass']:
            reward = -0.5
        elif possession['turnover']:
            reward = -1
        elif possession['violation']:
            reward = -1

        if possession['And1']:
            reward += self.compute_player_ft_reward(possession.player_name, shoots_num=1)
        return reward

    @staticmethod
    def compute_player_ft_reward(player_name, shoots_num):
        if player_name == 'n.hilário':
            player_name = 'n.hilario'
        player_stats = Codes.season_stats.loc[Codes.season_stats.pbp_name == player_name]
        if len(player_stats) == 0:
            ft_p = 0.757
        elif player_stats['FT%'] is None:
            ft_p = 0.757  # League average
        else:
            ft_p = player_stats['FT%'].values[0]
        if np.isnan(ft_p):
            ft_p = 0.757 # League average
        return ft_p * shoots_num

    def convert_moments_to_actions_and_states(self, moments, passes, is_attacking_the_left):
        possession_length = len(moments['locations'])
        states = self.create_states_data(moments, is_attacking_the_left)
        discrete_actions, continuous_actions = self.create_action_data(moments, passes, possession_length)
        return states, discrete_actions, continuous_actions

    def create_states_data(self, moments, is_attacking_the_left):

        player_in_possession = self.add_ball_handler(moments) # Add ball handler as part of the state

        quarter_clock = pd.Series(moments['quarter_clock'], name='quarter_clock').reset_index(drop=True)
        quarter_clock = quarter_clock / 720 # 60 seconds * 12 minutes

        shot_clock = pd.Series(moments['shot_clock'], name='shot_clock').reset_index(drop=True)
        shot_clock = shot_clock / 24 # Max seconds in a possession

        pass_ongoing = pd.Series(moments['pass_index'], name='pass_ongoing')

        locations = moments['locations']
        locations = locations.divide(self.divide_locations)

        is_attacking_the_left = pd.Series([is_attacking_the_left] * len(pass_ongoing))

        states = pd.concat([is_attacking_the_left, player_in_possession, quarter_clock, shot_clock, pass_ongoing, locations], axis=1)

        return states

    def add_ball_handler(self, moments):
        player_in_possession = pd.Series(moments['player_in_possession'], name='player_in_possession')
        player_in_possession = pd.get_dummies(player_in_possession)
        # Fix cases where some players didn't have possession of the ball in a certain possession
        possible_players_in_possession = [-1, 0, 1, 2, 3, 4]
        for possible_value in possible_players_in_possession:
            if possible_value not in player_in_possession.columns:
                player_in_possession[possible_value] = 0
        player_in_possession = player_in_possession.reindex(sorted(player_in_possession.columns), axis=1)
        return player_in_possession

    def create_action_data(self, moments, passes, possession_length):
        shots = pd.DataFrame.from_dict(moments['shots']).reset_index(drop=True)
        shots.columns = ['shot_index']
        passes_df = self.convert_pass_index_to_actions(passes, possession_length).reset_index(drop=True)

        # Used to be for moments['locations']
        movement_vectors = pd.DataFrame(moments['movement_vectors'])

        movement_vectors = self.get_relevant_team_locations(movement_vectors, 0)

        discrete_actions = pd.concat([shots, passes_df], axis=1)
        continuous_actions = movement_vectors

        curr_max_velocity = movement_vectors.abs().max().max()
        if self.max_velocity < curr_max_velocity:
            self.max_velocity = curr_max_velocity

        return discrete_actions, continuous_actions

    def get_relevant_team_locations(self, data, loc_start=0):

        if self.offense:
            if self.is_first_team_attacking:
                relevant_locations = data.iloc[:, loc_start:loc_start + 13]
            else:
                relevant_locations = pd.concat(
                    [data.iloc[:, loc_start:loc_start + 3], data.iloc[:, loc_start + 13:]], axis=1)
        else:
            # Defensive team movements
            if self.is_first_team_attacking:
                relevant_locations = data.iloc[:, loc_start + 13:]
            else:
                relevant_locations = data.iloc[:, loc_start + 3:loc_start + 13]

        return relevant_locations

    @staticmethod
    def get_players_velocities_deltas(moments):
        # Removing the first 3 locations in order to not include the ball's data
        movement_vectors = pd.DataFrame(moments['movement_vectors'])
        movement_vectors_deltas = list()
        previous_row = movement_vectors.iloc[0]
        for i, row in movement_vectors.iterrows():
            deltas = row.values[3:] - previous_row[3:]
            movement_vectors_deltas.append(deltas)
            previous_row = row
        movement_vectors_deltas = pd.DataFrame(movement_vectors_deltas).reset_index(drop=True)
        return movement_vectors_deltas

    @staticmethod
    def convert_pass_index_to_actions(passes, possession_length):
        passer_encoding = np.zeros((possession_length, 5))

        for i in range(possession_length):
            if str(i) in passes:
                pass_dict = passes[str(i)]
                passer, interceptor = pass_dict['players'][0], pass_dict['players'][1]
                pass_type = pass_dict['type']
                if pass_type != 'handoff':
                    pass_start_index = pass_dict['index'][0]
                else:
                    pass_start_index = pass_dict['index']

                passer_encoding[pass_start_index][interceptor] = 1

        pass_actions_df = pd.DataFrame(passer_encoding)
        columns = ['passer_' + str(i) for i in range(5)]
        pass_actions_df.columns = columns
        return pass_actions_df

    @staticmethod
    def transpose_xy(locations):
        for column_name, series in locations.iteritems():
            if 'x' in column_name:
                locations[column_name] = Codes.X_MAX - series
            elif 'y' in column_name:
                locations[column_name] = Codes.Y_MAX - series
        return locations

    def transpose_locations(self, game, i):
        # for i, possession in game.merged_data.iterrows():
        locations = game.merged_data.at[i, 'moments']['locations']
        x_values = list()
        for column in locations.columns:
            if 'x' in column:
                x_values = x_values + list(locations[column].values)

        transpose_locations = self.transpose_xy(locations)
        game.merged_data.at[i, 'moments']['locations'] = transpose_locations

    def get_episode_data(self):
        if hasattr(self, 'episodes'):
            if self.next_episode_id < len(self.episodes):
                episode = self.episodes[self.next_episode_id]
                self.next_episode_id += 1
                if len(episode['sequences']) > 0:
                    return episode['sequences']
                else:
                    return self.get_episode_data()
            elif self.next_pickle == len(self.pickle_list):
                    return None
            else:
                self.load_next_pickle()
                return self.get_episode_data()
        else:
            self.load_next_pickle()
            return self.get_episode_data()

    def load_next_pickle(self, verbose=False):
        next_pickle = self.pickle_list[self.next_pickle]
        pickle_path = os.path.join(self.episodes_save_path, next_pickle)
        with open(pickle_path, 'rb') as handle:
            self.episodes = pickle.load(handle)
        if verbose:
            print(f'Finished loading game {self.next_pickle}, this game contains {len(self.episodes)} episodes')
        self.next_pickle += 1
        self.next_episode_id = 0

    def load_specific_pickle(self, i):
        next_pickle = self.pickle_list[i]
        pickle_path = os.path.join(self.episodes_save_path, next_pickle)
        with open(pickle_path, 'rb') as handle:
            episodes = pickle.load(handle)
        return episodes

    @staticmethod
    def is_attacking_the_left_side(possession):
        locations = possession['moments']['locations']
        x_values = list()
        for column in locations.columns:
            if 'x' in column:
                x_values = x_values + list(locations[column].values)

        mean_x_mid = sum(x_values) / len(x_values)

        if mean_x_mid > 47:
            return 0
        else:
            return 1