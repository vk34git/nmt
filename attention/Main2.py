from AttentionModel import AttentionModel
from Data import Data

d=Data()
d.get_dataset()
m = AttentionModel(d)
m.load_model()
m.evaluate_model()

