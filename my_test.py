from ultra_50g.modeling import UltraForKnowledgeGraphReasoning
from ultra_50g.ultra.datasets import CoDExSmall
from ultra_50g.ultra.eval import test

model = UltraForKnowledgeGraphReasoning.from_pretrained("mgalkin/ultra_50g")
dataset = CoDExSmall(root="./datasets/")
test(model, mode="test", dataset=dataset, gpus=None)
print("Done!")
