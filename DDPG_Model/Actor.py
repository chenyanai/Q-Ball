from tensorflow.keras.layers import Dense, Input, Embedding, GRU, Concatenate, Flatten, Dropout, LSTM
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import Codes

class ActorNetwork(object):
    def __init__(self, state_dim, discrete_action_dim, continuous_action_dim, sequence_length=25):

        self.state_dim = state_dim
        # self.meta_data_dim = meta_data_dim
        self.sequence_length = sequence_length
        self.discrete_action_dim = discrete_action_dim
        self.continuous_action_dim = continuous_action_dim

    def create_actor_network(self):
        initializer = tf.keras.initializers.he_uniform()

        ### State Inputs:
        # meta_data_input = Input(shape=(self.meta_data_dim), name='state_input')
        state_input = Input(shape=(self.sequence_length, self.state_dim), name='state_input')

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
        players_embedding_layer = Embedding(len(Codes.players_dict) + 1, 10,
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

        ### Concatenate all the data:
        embeddings = Concatenate(axis=-1)(
        players_embeddings + [off_team_emb, def_team_emb, offensive_players_stats_input, defensive_players_stats_input])

        embedding_dense = Dense(32, activation='relu',
                                name='embedding_dense',
                                kernel_initializer=initializer,
                                kernel_regularizer=l2(0.01),
                                activity_regularizer=l2(0.01))(embeddings)

        ### GRU:
        gru_layer_seq = LSTM(50,
                             input_shape=(self.sequence_length,self.state_dim),
                             return_sequences=False,
                             activity_regularizer=l2(1e-5),
                             bias_regularizer=l2(0.01),
                             kernel_regularizer=l2(0.01),
                             recurrent_regularizer=l2(0.01))(state_input)

        concat_all = Concatenate(axis=-1)([embedding_dense, gru_layer_seq])

        ### All data dense layer
        dense = Dense(64, activation='relu',
                      name='first_dense',
                      kernel_initializer=initializer,
                      kernel_regularizer=l2(0.01),
                      activity_regularizer=l2(0.01))(concat_all)

        ### Outputs:
        discrete_actions_dense_output = Dense(self.discrete_action_dim,
                                              activation='softmax',
                                              name='discrete_output',
                                              kernel_initializer=initializer,
                                              kernel_regularizer=l2(0.01),
                                              activity_regularizer=l2(0.01))(dense)


        continuous_actions_dense_output = Dense(self.continuous_action_dim,
                                                activation='tanh',
                                                name='continuous_output',
                                                kernel_initializer=initializer,
                                                kernel_regularizer=l2(0.01),
                                                activity_regularizer=l2(0.01))(dense)

        # Model configuration:
        model = Model(players_inputs + [state_input, offensive_team_input, defensive_team_input,
                                        offensive_players_stats_input, defensive_players_stats_input],
                               [discrete_actions_dense_output, continuous_actions_dense_output, ])

        return model