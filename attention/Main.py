from AttentionModel import AttentionModel
from Data import Data


d = Data()
d.get_dataset()
m = AttentionModel(d)

# model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
# model.summary()

m.pipeline()
m.load_model()
m.evaluate_model()


