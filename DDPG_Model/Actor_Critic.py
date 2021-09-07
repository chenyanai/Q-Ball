from tensorflow.keras.layers import Dense, Input, Embedding, GRU, Concatenate, Flatten, Dropout, LSTM
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import Codes

class ActorCriticNetwork(object):
    def __init__(self, state_dim, discrete_action_dim, continuous_action_dim, sequence_length=25):

        self.state_dim = state_dim
        # self.meta_data_dim = meta_data_dim
        self.sequence_length = sequence_length
        self.discrete_action_dim = discrete_action_dim
        self.continuous_action_dim = continuous_action_dim

    def create_model(self, num_of_dense=1):
        initializer = tf.keras.initializers.he_uniform()

        ### State Inputs:
        # meta_data_input = Input(shape=(self.meta_data_dim), name='state_input')
        state_input = Input(shape=(self.sequence_length, self.state_dim), name='state_input')

        ### Embedding Inputs:
        offensive_team_input = Input(shape=(1, ), name='offensive_team_input')
        defensive_team_input = Input(shape=(1, ), name='defensive_team_input')

        off_team_players_inputs = list()
        for i in range(5):
            off_team_players_inputs.append(Input(shape=(1, ), name=('offensive_player_' + str(i))))

        def_team_players_inputs = list()
        for i in range(5):
            def_team_players_inputs.append(Input(shape=(1, ), name=('defensive_player_' + str(i))))

        players_inputs = off_team_players_inputs + def_team_players_inputs

        ### Team and players embedding:
        players_embedding_layer = Embedding(len(Codes.players_dict) + 1, 3,
                                            name='players_embedding',
                                            embeddings_initializer=initializer,
                                            activity_regularizer=l2(0.01),
                                            embeddings_regularizer=l2(0.01))

        players_embeddings = list()
        for player_input in players_inputs:
            players_embeddings.append(Flatten()(players_embedding_layer(player_input)))

        teams_embedding_layer = Embedding(33, 3,
                                          name='teams_embedding',
                                          embeddings_initializer=initializer,
                                          activity_regularizer=l2(0.01))

        off_team_emb = Flatten()(teams_embedding_layer(offensive_team_input))
        def_team_emb = Flatten()(teams_embedding_layer(defensive_team_input))

        ### Concatenate all the data:
        embeddings = Concatenate(axis=-1)(
            players_embeddings + [off_team_emb, def_team_emb])

        embedding_dense = Dense(8, activation='relu',
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

        concat_all_actor = Concatenate(axis=-1)([embedding_dense, gru_layer_seq])

        ### All data dense layer
        dense = Dense(64, activation='relu',
                      name='actor_first_dense',
                      kernel_initializer=initializer,
                      kernel_regularizer=l2(0.01),
                      activity_regularizer=l2(0.01))(concat_all_actor)

        if num_of_dense > 1:
            dense = Dense(64, activation='relu',
                          name='actor_last_dense',
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(0.01),
                          activity_regularizer=l2(0.01))(dense)

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

        ### Critic ###

        ### Concatenate all the data:
        concat_all_critic = \
            Concatenate(axis=-1, name='states_and_actions')\
                ([gru_layer_seq, discrete_actions_dense_output, continuous_actions_dense_output, embedding_dense])

        ## All data dense layer
        if num_of_dense > 1:
            activation = 'relu'
        else:
            activation = 'tanh'

        dense = Dense(64, activation=activation,
                      name='critic_first_dense',
                      kernel_initializer=initializer,
                      kernel_regularizer=l2(0.01),
                      activity_regularizer=l2(0.01))(concat_all_critic)

        if num_of_dense > 1:
            dense = Dense(64, activation='tanh',
                          name='critic_last_dense',
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(0.01),
                          activity_regularizer=l2(0.01))(dense)

        ### Outputs:
        critic_value = Dense(1,
                             activation='linear',
                             name='q_value_output',
                             kernel_initializer=initializer,
                             activity_regularizer=l2(0.01),
                             kernel_regularizer=l2(0.01),
                             bias_regularizer=l2(0.01))(dense)

        # Model configuration:
        model = Model(inputs=players_inputs + [state_input, offensive_team_input, defensive_team_input],
                      outputs=[discrete_actions_dense_output, continuous_actions_dense_output, critic_value])

        return model

    def create_offense_critic_model(self, num_of_dense=1):
        initializer = tf.keras.initializers.he_uniform()

        ### State Inputs:
        # meta_data_input = Input(shape=(self.meta_data_dim), name='state_input')
        state_input = Input(shape=(self.sequence_length, self.state_dim), name='state_input')

        ### Embedding Inputs:
        offensive_team_input = Input(shape=(1, ), name='offensive_team_input')
        defensive_team_input = Input(shape=(1, ), name='defensive_team_input')

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
            players_embeddings + [off_team_emb, def_team_emb])

        embedding_dense = Dense(16, activation='relu',
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

        discrete_actions_input = \
            Input(shape=(self.discrete_action_dim,), name='discrete_action_input')
        continuous_actions_input = \
            Input(shape=(self.continuous_action_dim,), name='continuous_action_input')

        ### Critic ###

        ### Concatenate all the data:
        concat_all_critic = \
            Concatenate(axis=-1, name='states_and_actions')\
                ([gru_layer_seq, discrete_actions_input, continuous_actions_input, embedding_dense])

        ## All data dense layer
        if num_of_dense > 1:
            activation = 'relu'
        else:
            activation = 'tanh'

        dense = Dense(64, activation=activation,
                      name='critic_first_dense',
                      kernel_initializer=initializer,
                      kernel_regularizer=l2(0.01),
                      activity_regularizer=l2(0.01))(concat_all_critic)

        if num_of_dense > 1:
            dense = Dense(64, activation='tanh',
                          name='critic_last_dense',
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(0.01),
                          activity_regularizer=l2(0.01))(dense)

        ### Outputs:
        critic_value = Dense(1,
                             activation='linear',
                             name='q_value_output',
                             kernel_initializer=initializer,
                             activity_regularizer=l2(0.01),
                             kernel_regularizer=l2(0.01),
                             bias_regularizer=l2(0.01))(dense)

        # Model configuration:
        model = Model(inputs=players_inputs + [state_input, offensive_team_input, defensive_team_input,
                                               discrete_actions_input, continuous_actions_input],
                      outputs=[critic_value])

        return model

    def load_offense_critic(self, model_path, num_of_dense=1):
        model = self.create_offense_critic_model(num_of_dense)
        model.load_weights(model_path, by_name=True)
        return model

    def create_defensive_agent_model(self, num_of_dense=1):
        initializer = tf.keras.initializers.he_uniform()

        ### State Inputs:
        state_input = Input(shape=(self.sequence_length, self.state_dim + self.discrete_action_dim), name='state_input')

        ### Embedding Inputs:
        offensive_team_input = Input(shape=(1,), name='offensive_team_input')
        defensive_team_input = Input(shape=(1,), name='defensive_team_input')

        off_team_players_inputs = list()
        for i in range(5):
            off_team_players_inputs.append(Input(shape=(1,), name=('offensive_player_' + str(i))))

        def_team_players_inputs = list()
        for i in range(5):
            def_team_players_inputs.append(Input(shape=(1,), name=('defensive_player_' + str(i))))

        players_inputs = off_team_players_inputs + def_team_players_inputs

        ### Team and players embedding:
        players_embedding_layer = Embedding(len(Codes.players_dict) + 1, 3,
                                            name='players_embedding',
                                            embeddings_initializer=initializer,
                                            activity_regularizer=l2(0.01),
                                            embeddings_regularizer=l2(0.01))

        players_embeddings = list()
        for player_input in players_inputs:
            players_embeddings.append(Flatten()(players_embedding_layer(player_input)))

        teams_embedding_layer = Embedding(33, 3,
                                          name='teams_embedding',
                                          embeddings_initializer=initializer,
                                          activity_regularizer=l2(0.01))

        off_team_emb = Flatten()(teams_embedding_layer(offensive_team_input))
        def_team_emb = Flatten()(teams_embedding_layer(defensive_team_input))

        ### Concatenate all the data:
        embeddings = Concatenate(axis=-1)(
            players_embeddings + [off_team_emb, def_team_emb])

        embedding_dense = Dense(8, activation='relu',
                                name='embedding_dense',
                                kernel_initializer=initializer,
                                kernel_regularizer=l2(0.01),
                                activity_regularizer=l2(0.01))(embeddings)

        ### GRU:
        gru_layer_seq = LSTM(50,
                             input_shape=(self.sequence_length, self.state_dim + self.discrete_action_dim),
                             return_sequences=False,
                             activity_regularizer=l2(1e-5),
                             bias_regularizer=l2(0.01),
                             kernel_regularizer=l2(0.01),
                             recurrent_regularizer=l2(0.01))(state_input)

        concat_all_actor = Concatenate(axis=-1)([embedding_dense, gru_layer_seq])

        ### All data dense layer
        dense = Dense(64, activation='relu',
                      name='actor_first_dense',
                      kernel_initializer=initializer,
                      kernel_regularizer=l2(0.01),
                      activity_regularizer=l2(0.01))(concat_all_actor)

        if num_of_dense > 1:
            dense = Dense(64, activation='relu',
                          name='actor_last_dense',
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(0.01),
                          activity_regularizer=l2(0.01))(dense)

        ### Output:
        continuous_actions_dense_output = Dense(self.continuous_action_dim,
                                                activation='tanh',
                                                name='continuous_output',
                                                kernel_initializer=initializer,
                                                kernel_regularizer=l2(0.01),
                                                activity_regularizer=l2(0.01))(dense)

        ### Critic ###

        ### Concatenate all the data:
        concat_all_critic = \
            Concatenate(axis=-1, name='states_and_actions') \
                ([gru_layer_seq, continuous_actions_dense_output, embedding_dense])

        ### All data dense layer
        if num_of_dense > 1:
            activation = 'relu'
        else:
            activation = 'tanh'

        dense = Dense(64, activation=activation,
                      name='critic_first_dense',
                      kernel_initializer=initializer,
                      kernel_regularizer=l2(0.01),
                      activity_regularizer=l2(0.01))(concat_all_critic)

        if num_of_dense > 1:
            dense = Dense(64, activation='tanh',
                          name='critic_last_dense',
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(0.01),
                          activity_regularizer=l2(0.01))(concat_all_critic)

        ### Outputs:
        critic_value = Dense(1,
                             activation='linear',
                             name='q_value_output',
                             kernel_initializer=initializer,
                             activity_regularizer=l2(0.01),
                             kernel_regularizer=l2(0.01),
                             bias_regularizer=l2(0.01))(dense)

        # Model configuration:
        model = Model(inputs=players_inputs + [state_input, offensive_team_input, defensive_team_input],
                      outputs=[continuous_actions_dense_output, critic_value])

        return model