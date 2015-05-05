import unittest
from est import *
import numpy as np

class TestBS_quad(unittest.TestCase):

	def test_BS_notmatrix(self):
		self.assertRaises(AttributeError, getattr,0,"BS_quad")

	def test_BS_ones(self):
		self.assertAlmostEqual(BS_quad(np.ones((8,8))),0)

	def test_BS_identity(self):
		self.assertAlmostEqual(BS_quad(np.identity(4)),1.0/24)
		#Need almost equal here and not for the other two.

	def test_BS_lowertri(self):
		lt = np.array([[1,0,0,0,0,0,0,0],[1,1,0,0,0,0,0,0],[1,1,1,0,0,0,0,0],[1,1,1,1,0,0,0,0],[1,1,1,1,1,0,0,0],[1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0],[1,1,1,1,1,1,1,1]])
		self.assertAlmostEqual(BS_quad(lt),153.0/320)

	def test_BS_nonsquare(self):
		mat = np.array([[1,0,0,1,0,0,1],[1,0,0,0,0,0,0],[1,1,1,1,1,1,1],[1,1,1,1,1,0,0],[1,1,0,1,0,1,0]])
		self.assertAlmostEqual(BS_quad(mat),929.0/9261)
		#Not sure why I need almost equal here and not for the other two.

class TestCQ_quad(unittest.TestCase):

	def test_CQ_notmatrix(self):
		self.assertRaises(AttributeError, getattr,0,"CQ_quad")

	def test_CQ_ones(self):
		self.assertAlmostEqual(CQ_quad(np.ones((8,8))),0)

	def test_CQ_identity(self):
		self.assertAlmostEqual(CQ_quad(np.identity(4)),0)

	def test_cQ_lowertri(self):
		lt = np.array([[1,0,0,0,0],[1,1,0,0,0],[1,1,1,0,0],[1,1,1,1,0],[1,1,1,1,1]])
		self.assertAlmostEqual(CQ_quad(lt),43.0/90)



	def test_CQ_nonsquare(self):
		mat = np.array([[1,0,0,1,0,0,1],[1,0,0,0,0,0,0],[1,1,1,1,1,1,1],[1,1,1,1,1,0,0],[1,1,0,1,0,1,0]])
		self.assertAlmostEqual(CQ_quad(mat),34.0/175)


class TestThreshold_quad(unittest.TestCase):

	def test_threshold_notmatrix(self):
		self.assertRaises(AttributeError, getattr,0,"threshold_quad")

	def test_threshold_identity(self):
		self.assertItemsEqual(threshold_quad(np.identity(4),0),[0,0,0])
		self.assertItemsEqual(threshold_quad(np.identity(4),1),[0,0,0])
		self.assertItemsEqual(threshold_quad(np.identity(4),1.5),[0,0,0])


	def test_CQ_ones(self):
		self.assertItemsEqual(threshold_quad(np.ones((8,8)),0),[56,8,64])
		self.assertItemsEqual(threshold_quad(np.ones((8,8)),1),[56,8,64])
		self.assertItemsEqual(threshold_quad(np.ones((8,8)),1.5),[56,8,64])


	def test_threshold_lowertri(self):
		lt = np.array([[1,0,0,0,0],[1,1,0,0,0],[1,1,1,0,0],[1,1,1,1,0],[1,1,1,1,1]])
		self.assertItemsEqual(threshold_quad(lt,0),[4,2,6])
		self.assertAlmostEqual(threshold_quad(lt,1)[0],68.0/25)
		self.assertAlmostEqual(threshold_quad(lt,1)[1],2)
		self.assertAlmostEqual(threshold_quad(lt,1)[2],118.0/25)		
		self.assertItemsEqual(threshold_quad(lt,1.5),[0,2,2])

	def test_threshold_lowertri_montonicity(self):
		lt = np.array([[1,0,0,0,0],[1,1,0,0,0],[1,1,1,0,0],[1,1,1,1,0],[1,1,1,1,1]])
		self.assertEqual((threshold_quad(lt,0)[0]>threshold_quad(lt,1)[0] and threshold_quad(lt,1)[0]>threshold_quad(lt,1.5)[0]),True)
		self.assertEqual((threshold_quad(lt,0)[1]==threshold_quad(lt,1)[1] and threshold_quad(lt,1)[1]==threshold_quad(lt,1.5)[1]),True)
		self.assertEqual((threshold_quad(lt,0)[2]>threshold_quad(lt,1)[2] and threshold_quad(lt,1)[2]>threshold_quad(lt,1.5)[2]),True)



	def test_threshold_nonsquare(self):
		mat = np.array([[1,0,0,1,0,0,1],[1,0,0,0,0,0,0],[1,1,1,1,1,1,1],[1,1,1,1,1,0,0],[1,1,0,1,0,1,0]])
		self.assertAlmostEqual(threshold_quad(mat,0)[0],142.0/49)
		self.assertAlmostEqual(threshold_quad(mat,0)[1],40.0/21)
		self.assertAlmostEqual(threshold_quad(mat,0)[2],706.0/147)

		self.assertAlmostEqual(threshold_quad(mat,1)[0],82.0/49)
		self.assertAlmostEqual(threshold_quad(mat,1)[1],40.0/21)
		self.assertAlmostEqual(threshold_quad(mat,1)[2],526.0/147)

		self.assertAlmostEqual(threshold_quad(mat,2.5)[0],0)
		self.assertAlmostEqual(threshold_quad(mat,2.5)[1],40.0/21)
		self.assertAlmostEqual(threshold_quad(mat,2.5)[2],40.0/21)


	def test_threshold_nonsquare_montonicity(self):
		mat = np.array([[1,0,0,1,0,0,1],[1,0,0,0,0,0,0],[1,1,1,1,1,1,1],[1,1,1,1,1,0,0],[1,1,0,1,0,1,0]])
		self.assertEqual((threshold_quad(mat,0)[0]>threshold_quad(mat,1)[0] and threshold_quad(mat,1)[0]>threshold_quad(mat,1.5)[0]),1)
		self.assertEqual((threshold_quad(mat,0)[1]==threshold_quad(mat,1)[1] and threshold_quad(mat,1)[1]==threshold_quad(mat,1.5)[1]),1)
		self.assertEqual((threshold_quad(mat,0)[2]>threshold_quad(mat,1)[2] and threshold_quad(mat,1)[2]>threshold_quad(mat,1.5)[2]),1)



if __name__ == "__main__":
	unittest.main()