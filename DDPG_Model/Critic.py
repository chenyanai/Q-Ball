from tensorflow.keras.layers import Dense, Input, Embedding, GRU, Concatenate, Flatten, Dropout, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from Codes import players_dict
import tensorflow as tf

class CriticNetwork(object):
    def __init__(self, state_dim, discrete_action_dim, continuous_action_dim, sequence_length=25):

        self.sequence_length = sequence_length
        self.state_dim = state_dim
        # self.meta_data_dim = meta_data_dim
        self.discrete_action_dim = discrete_action_dim
        self.continuous_action_dim = continuous_action_dim

    def create_critic_network(self):
        initializer = tf.keras.initializers.he_uniform()

        ### State Inputs:
        state_input = Input(shape=(self.sequence_length, self.state_dim), name='state_input')

        ### Action Inputs:
        discrete_actions_input = \
            Input(shape=(self.discrete_action_dim, ), name='discrete_action_input')
        continuous_actions_input = \
            Input(shape=(self.continuous_action_dim, ), name='continuous_action_input')

        ### Embedding Inputs:
        offensive_team_input = Input(shape=(1, ), name='offensive_team_input')
        defensive_team_input = Input(shape=(1, ), name='defensive_team_input')

        offensive_players_stats_input = Input(shape=(50, ), name='offensive_players_stats_input')
        defensive_players_stats_input = Input(shape=(35, ), name='defensive_players_stats_input')

        off_team_players_inputs = list()
        for i in range(5):
            off_team_players_inputs.append(Input(shape=(1, ), name=('offensive_player_' + str(i))))

        def_team_players_inputs = list()
        for i in range(5):
            def_team_players_inputs.append(Input(shape=(1, ), name=('defensive_player_' + str(i))))

        players_inputs = off_team_players_inputs + def_team_players_inputs

        ### Team and players embedding:
        players_embedding_layer = Embedding(len(players_dict) + 1, 10,
                                            name='players_embedding',
                                            embeddings_initializer=initializer,
                                            activity_regularizer=l2(0.01),
                                            embeddings_regularizer=l2(0.01))

        players_embeddings = list()
        for player_input in players_inputs:
            players_embeddings.append(Flatten()(players_embedding_layer(player_input)))

        teams_embedding_layer = Embedding(33, 10,
                                          name='teams_embedding',
                                          embeddings_initializer=initializer,
                                          activity_regularizer=l2(0.01))

        off_team_emb = Flatten()(teams_embedding_layer(offensive_team_input))
        def_team_emb = Flatten()(teams_embedding_layer(defensive_team_input))

        ### Concatenate and send through dense layer all the embeddings
        embeddings = Concatenate(axis=-1)(
        players_embeddings + [off_team_emb, def_team_emb, offensive_players_stats_input, defensive_players_stats_input])

        embedding_dense = Dense(8, activation='relu',
                                name='embedding_dense',
                                kernel_initializer=initializer,
                                kernel_regularizer=l2(0.01),
                                activity_regularizer=l2(0.01))(embeddings)

        ### GRU:
        gru_layer_seq = LSTM(50,
                             input_shape=(self.sequence_length, self.state_dim),
                             return_sequences=False,
                             activity_regularizer=l2(1e-5),
                             bias_regularizer=l2(0.01),
                             kernel_regularizer=l2(0.01),
                             recurrent_regularizer=l2(0.01))(state_input)

        ### Concatenate all the data:
        concat_all = \
            Concatenate(axis=-1, name='states_and_actions')\
                ([gru_layer_seq, discrete_actions_input, continuous_actions_input, embedding_dense])

        ### All data dense layer
        dense = Dense(64, activation='tanh',
                      name='first_dense',
                      kernel_initializer=initializer,
                      kernel_regularizer=l2(0.01),
                      activity_regularizer=l2(0.01))(concat_all)

        ### Outputs:
        critic_value = Dense(1,
                              activation='linear',
                              name='discrete_output',
                              kernel_initializer=initializer,
                              activity_regularizer=l2(0.01),
                              kernel_regularizer=l2(0.01),
                              bias_regularizer=l2(0.01))(dense)


        # Model configuration:
        model = Model(inputs=[state_input, discrete_actions_input, continuous_actions_input, offensive_team_input,
                              defensive_team_input, offensive_players_stats_input, defensive_players_stats_input]
                             + players_inputs,
                      outputs=critic_value)
        return model