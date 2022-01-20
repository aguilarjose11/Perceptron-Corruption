from Perceptron.perceptron import PocketPerceptron
from tests.conftest import xor_perceptron

class TestPocketXor:
    def test_xor_unin_perceptron(self, xor_perceptron):
        """Test on uninitialized model"""
        algo = PocketPerceptron(
            input       =3,
            eta         =1,
            max_iter    =1000,
            rand_seed   =37
        )

        for X in xor_perceptron["X"]:
            assert algo.solve(X)

        del algo

    def test_xor_opt_perceptron(self, xor_perceptron):
        """Test on optimal model"""
        algo = PocketPerceptron(
            input       =3,
            eta         =1,
            max_iter    =1000,
            rand_seed   =37
        )
        algo.W = xor_perceptron["optimal_w"]

        for X, y in zip(xor_perceptron["X"], xor_perceptron["optimal_y"]):
            assert algo.solve(X) == y

        del algo