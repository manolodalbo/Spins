import unittest
import numpy as np
from preprocess import fm
class Test_FM(unittest.TestCase):
    def test_continuous(self):
        upper_freq = 5e9
        lower_freq = 3e9
        dt = 20e-12
        t = np.arange(0,600*dt,dt)
        high = 2 * np.sin(2*np.pi*upper_freq*t)
        high = np.expand_dims(high,axis=0)
        low = 0.5 * np.sin(2*np.pi*lower_freq*t)
        low = np.expand_dims(low,axis=0)
        ones = np.ones((1,600))
        zeros = np.zeros((1,600))
        modulated_high = fm(ones,lower_freq,upper_freq)
        modulated_low = fm(zeros,lower_freq,upper_freq)
        print(modulated_low[0,0:10])
        print(low[0,0:10])
        self.assertTrue(np.allclose(high,modulated_high,rtol=0.01))
        self.assertTrue(np.allclose(low,modulated_low,rtol=0.01))
if __name__ == '__main__':
    unittest.main()