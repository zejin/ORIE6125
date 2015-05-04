import unittest
from est import *
import numpy as np

class TestBS_quad(unittest.TestCase):
	def test_nonmatrix(self):
		self.assertRaises(AttributeError, getattr,0,"BS_quad")

	def test_identity(self):
		self.assertAlmostEqual(BS_quad(np.identity(4)),1.0/24)
		#Not sure why I need almost equal here and not for the other two.

	def test_lowertri(self):
		lt = np.array([[1,0,0,0,0,0,0,0],[1,1,0,0,0,0,0,0],[1,1,1,0,0,0,0,0],[1,1,1,1,0,0,0,0],[1,1,1,1,1,0,0,0],[1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0],[1,1,1,1,1,1,1,1]])
		self.assertEqual(BS_quad(lt),153.0/320)

	def test_ones(self):
		self.assertEqual(BS_quad(np.ones((8,8))),0)

	def test_nonsquare(self):
		mat = np.array([[1,0,0,1,0,0,1],[1,0,0,0,0,0,0],[1,1,1,1,1,1,1],[1,1,1,1,1,0,0],[1,1,0,1,0,1,0]])
		self.assertAlmostEqual(BS_quad(mat),929.0/9261)
		#Not sure why I need almost equal here and not for the other two.




if __name__ == "__main__":
	unittest.main()