from nmt_utils import *


class Data:

    m = 10000
    Tx = 30
    Ty = 10

# load data
    def get_dataset(self):

        dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(self.m)

        X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, self.Tx, self.Ty)

        print("X.shape:", X.shape)
        print("Y.shape:", Y.shape)
        print("Xoh.shape:", Xoh.shape)
        print("Yoh.shape:", Yoh.shape)

        self.dataset, self.human_vocab, self.machine_vocab, self.inv_machine_vocab = dataset, human_vocab, machine_vocab, inv_machine_vocab
        self.X, self.Y, self.Xoh, self.Yoh = X, Y, Xoh, Yoh


    def test(self):
        pass

