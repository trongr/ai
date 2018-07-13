import unittest
import numpy as np
import utils


class TestUtils(unittest.TestCase):

    # NOTE. Test methods have to start with test_
    def test_GenerateSimilarEncodings(self):
        encoding = [
            -0.35367707888908284, 0.621859179804771, -0.31997089483427876,
            0.011595342768606631, 0.4519630935068115, 0.2953935067217639,
            -0.8588470203851124, 0.8933864632984607, 0.4165123209785375,
            -0.38113172744101687, -0.7517159099074882, 0.4011720228973874,
            0.8851441012741836, 0.9942635625424923, 0.383328340500354,
            -0.7452014121765651, 0.8201744706484477, 0.8276165651149492,
            0.9946293171042713, 0.353367171493874, 0.22125994171277585,
            -0.7736084004873831, 0.5542256044536293, 0.005103391965230797,
            -0.8243577553047967, -0.356431396835301, -0.7448133054332053,
            0.5144217084975458, -0.2941842297947532, -0.573527983108516,
            0.04761633631126205, -0.4878006464493789, -0.395356525705417,
            -0.3415650931489096, -0.8382277229277855, 0.2457875463560928,
            -0.9435509270750619, -0.4696083953753343, 0.49697000256785984,
            -0.9444866003079233, -0.4326005790765919, -0.674353019990334,
            0.584421652073511, -0.8922725668803133, 0.03924468313166485,
            -0.08130156654849485, 0.34348350361228275, 0.9471490260846529,
            0.33509804732600323, -0.7826974079197473, -0.9828226176754866,
            0.05126473104450513, 0.468536304151026, -0.08998477098474766,
            0.05737140792290174, 0.16984371407441867, -0.18112497377841197,
            -0.871034716456454, 0.4032662404435483, -0.2755849575869713,
            -0.895811743375795, -0.22047261036707666, 0.4393597079051237,
            0.6809306748047443
        ]
        count = 100
        std = 0.2
        FreeChannels = 64
        encodings = utils.GenerateSimilarEncodings(encoding, std, FreeChannels,
                                                   count)
        self.assertEqual(len(encoding), 64)
        self.assertTrue(len(encodings), count)
        self.assertTrue(len(encodings[99]), len(encoding))
        self.assertTrue(np.max(encodings) <= 1)
        self.assertTrue(np.min(encodings) >= -1)
        self.assertTrue(np.array_equal(encodings[0], encoding))

    def test_GenerateChannelNoise(self):
        height = 8
        width = 8
        std = 0.2
        FreeChannels = 4
        noise = utils.GenerateChannelNoise(std, height, width, FreeChannels)
        self.assertEqual(noise.shape, (width, height))
        self.assertEqual(np.count_nonzero(noise), 4 * height)


if __name__ == '__main__':
    unittest.main()
