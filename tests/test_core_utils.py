import unittest
import torch
import torch.nn.functional as F
from src.utils.core_utils import fpr_from_logits

class TestCoreUtils(unittest.TestCase):
    def test_fpr_from_logits(self):
        logits = torch.randn(5, 2)
        probs = F.softmax(logits)
        targets = torch.tensor([0, 1, 0, 0, 1], dtype=torch.int64)
        print("logits:", logits)
        print("probs:", probs)
        print("targets:", targets)

        fpr, fpr_proxy = fpr_from_logits(logits, targets)
        print("fpr:", fpr)
        print("fpr proxy:", fpr_proxy)

def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(TestCoreUtils('test_fpr_from_logits'))
    return test_suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())