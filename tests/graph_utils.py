import unittest
import torch
from src.utils.graph_utils import subgraph_negative_sampling

class TestGraphUtils(unittest.TestCase):
    def test_subgraph_negative_sampling(self):
        edge_index = torch.tensor([[0,0,1,1,3,4,4,5],
                                   [1,5,0,4,4,1,3,0]], dtype=torch.int)
        subgraph_mask = torch.tensor([0,0,1,1,1,1], dtype=torch.bool)
        negative_samples = subgraph_negative_sampling(edge_index, subgraph_mask)
        print(negative_samples)
        negative_samples = subgraph_negative_sampling(edge_index, subgraph_mask, num_neg_samples=14)
        print(negative_samples)

def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(TestGraphUtils('test_subgraph_negative_sampling'))
    return test_suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())