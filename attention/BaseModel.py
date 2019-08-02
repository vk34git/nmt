"""In this section, we will develop a baseline in performance on the problem with an encoder-decoder model without attention.

We will fix the problem definition at input and output sequences of 5 time steps, the first 2 elements of the input sequence in the output sequence and a cardinality of 50."""

from Data import Data
import numpy as np
from keras.models import model_from_json
from keras.models import load_model


from nmt_utils import *
class BaseModel:

    def __init__(self, d: Data):
        self.n_a = 32
        self.n_s = 64

        self.d = d

    def design_model(self):
        pass
    def fit_model(self):
        # train LSTM
        d = self.d
        model = self.model
        s0 = np.zeros((d.m, self.n_s))
        c0 = np.zeros((d.m, self.n_s))
        outputs = list(d.Yoh.swapaxes(0,1))
        model.fit([d.Xoh, s0, c0], outputs, epochs=10, batch_size=100)
        self.s0 = s0
        self.c0 = c0

    def evaluate_model(self):
        # evaluate LSTM
        d = self.d
        model = self.model
        EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001', 'March 3rd 2001', '1 March 2001']
        for example in EXAMPLES:

            source = string_to_int(example, d.Tx, d.human_vocab)
            print (len(source))
            print(len(d.human_vocab))
            source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(d.human_vocab)), source)))\
                # .swapaxes(0,1)
            source = np.reshape(source, (1, source.shape[0], source.shape[1]))


            s0 = np.zeros((d.m, self.n_s))
            c0 = np.zeros((d.m, self.n_s))

            print (source.shape)
            print("line 48:", s0, c0)
            prediction = model.predict([source, s0, c0])
            prediction = np.argmax(prediction, axis = -1)
            output = [d.inv_machine_vocab[int(i)] for i in prediction]

            print("source:", example)
            print("output:", ''.join(output))
# if array_equal(d.one_hot_decode(y[0]), d.one_hot_decode(yhat[0])):
#                 correct += 1
#         print('Accuracy: %.2f%%' % (float(correct)/float(total)*100.0))
        # spot check some examples

    def pipeline(self):
        self.design_model()
        self.fit_model()
        #self.evaluate_model()
        self.save_model()

    def save_model(self):

        self.model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
        del self.model  # deletes the existing model

    def save_model2(self):
        self.model.save_weights("model.h5")
        # Save the model architecture
        with open('model_architecture.json', 'w') as f:
            f.write(self.model.to_json())

        import pickle
        with open('context.pickle', 'wb') as f:
            pickle.dump([self.s0, self.c0], f)

    def load_model(self):
        # returns a compiled model
        # identical to the previous one
        self.model = load_model('my_model.h5')

    def load_model2(self):

        # Model reconstruction from JSON file
        with open('model_architecture.json', 'r') as f:
            model = model_from_json(f.read())

        # Load weights into the new model
        model.load_weights('model.h5')
        import pickle


        d = Data()
        d.get_dataset()
        m = BaseModel(d)
        m.model = model
        with open('context.pickle', 'rb') as f:
            [m.s0, m.c0] =  pickle.load(f)
        print (m.s0, m.c0)
        import keras
        opt = keras.optimizers.adam(lr = 0.005, beta_1 = 0.9, beta_2 = 0.999, decay = 0.01)
        m.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
