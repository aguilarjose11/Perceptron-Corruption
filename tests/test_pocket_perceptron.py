from Perceptron.perceptron import PocketPerceptron
from tests.conftest import xor_perceptron

import numpy as np

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

class TestPocketTraining:
    def test_learn(self, learn_xor_pocket):
        algo = PocketPerceptron(
            input       =3,
            eta         =1,
            max_iter    =1000,
            rand_seed   =37
        )

        algo.learn(learn_xor_pocket["X"], learn_xor_pocket["y"])

        assert algo.pi == learn_xor_pocket["pi"]
        assert algo.W == learn_xor_pocket["W"]
        assert algo.run_pi == learn_xor_pocket["run_pi"]
        assert algo.run_W == learn_xor_pocket["run_W"]
        
    def test_train(self, xor_perceptron):
        algo = PocketPerceptron(
            input       =3,
            eta         =1,
            max_iter    =1000,
            rand_seed   =37
        )

        algo.train(xor_perceptron["X"], xor_perceptron["y"])

        assert algo.W != np.zeros((3, 1))