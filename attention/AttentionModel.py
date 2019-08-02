from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.models import Model
from keras.optimizers import adam
from nmt_utils import *
from BaseModel import BaseModel
from Data import Data
class AttentionModel(BaseModel):

    repeator = RepeatVector(Data.Tx)
    concatenator = Concatenate(axis=-1)
    densor1 = Dense(10, activation = "tanh")
    densor2 = Dense(1, activation = "relu")
    activator = Activation(softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
    dotor = Dot(axes = 1)

    def one_step_attention(self, a, s_prev, Tx, Ty):
        """
        Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
        "alphas" and the hidden states "a" of the Bi-LSTM.

        Arguments:
        a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
        s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)

        Returns:
        context -- context vector, input of the next (post-attetion) LSTM cell
        """


        #print("a.shape", ' ', a.shape)
        #print("s_prev.shape", ' ', s_prev.shape)

          ### START CODE HERE ###
        # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)
        s_prev = AttentionModel.repeator(s_prev)
        # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)


        concat = AttentionModel.concatenator([a, s_prev])
        # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e. (≈1 lines)
        e = AttentionModel.densor1(concat)
        # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies. (≈1 lines)
        energies = AttentionModel.densor2(e)
        # Use "activator" on "energies" to compute the attention weights "alphas" (≈ 1 line)
        alphas = AttentionModel.activator(energies)
        # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)
        context = AttentionModel.dotor([alphas,a])
        ### END CODE HERE ###

        return context

    def design_model(self):
        """
        Arguments:
        Tx -- length of the input sequence
        Ty -- length of the output sequence
        n_a -- hidden state size of the Bi-LSTM
        n_s -- hidden state size of the post-attention LSTM
        human_vocab_size -- size of the python dictionary "human_vocab"
        machine_vocab_size -- size of the python dictionary "machine_vocab"

        Returns:
        model -- Keras model instance
        """

        # Define the inputs of your model with a shape (Tx,)
        # Define s0 and c0, initial hidden state for the decoder LSTM of shape (n_s,)
        post_activation_LSTM_cell = LSTM(self.n_s, return_state = True)
        output_layer = Dense(len(self.d.machine_vocab), activation=softmax)

        X = Input(shape=(self.d.Tx, len(self.d.human_vocab)))
        s0 = Input(shape=(self.n_s,), name='s0')
        c0 = Input(shape=(self.n_s,), name='c0')
        s = s0
        c = c0

        # Initialize empty list of outputs
        outputs = []

        ### START CODE HERE ###

        # Step 1: Define your pre-attention Bi-LSTM. Remember to use return_sequences=True. (≈ 1 line)

        Tx = self.d.Tx
        Ty = self.d.Ty
        m = self.d.m
        n_a = self.n_a

        a = Bidirectional(LSTM(n_a, return_sequences = True), input_shape = (m, Tx, n_a*2))(X)

        # Step 2: Iterate for Ty steps
        for t in range(Ty):

            # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t (≈ 1 line)
            context = self.one_step_attention(a, s, Tx, Ty)

            # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
            # Don't forget to pass: initial_state = [hidden state, cell state] (≈ 1 line)
            s, _, c = post_activation_LSTM_cell(context,initial_state = [s, c])

            # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
            out = output_layer(s)

            # Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
            outputs.append(out)

        # Step 3: Create model instance taking three inputs and returning the list of outputs. (≈ 1 line)
        model = Model([X, s0, c0], outputs = outputs)

        opt = adam(lr = 0.005, beta_1 = 0.9, beta_2 = 0.999, decay = 0.01)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

        self.model = model
