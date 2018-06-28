import numpy as np
import tensorflow as tf
import DeconvGConvDVanillaGAN as GAN
import utils
import MathLib


def TestRangeEachDimension():
    """Generates 64 (noise_dim) images, each image contains 100 faces, each a variation from -1 to 1 of the same noise dimension."""
    batch_size = 100
    noise_dim = 64
    noise_input = [8.685266433488878501e-01, -7.861549933695841652e-01, -6.398974390956200242e-01, 3.079600179814192540e-01, 6.314542425683029592e-01, -3.832037598255819688e-01, -6.867196305925225008e-01, -6.538568355329785753e-01, 4.467145970933827925e-01, 1.074893006118486927e-01, -4.184085090469162882e-01, -9.668555709334403225e-01, -7.277596303315085891e-01, 7.232803008497166175e-01, -4.118989412568132558e-01, 7.643974076799808781e-01, 6.837788739535317628e-01, -3.154805591013811750e-01, 9.793365723614095852e-01, 1.570314484185821868e-01, 7.243522191672078225e-01, -9.833831630156151249e-01, -5.808857773322504414e-01, -6.119468805001466727e-01, 6.989781669440628953e-01, -4.548371455960735776e-01, 4.337164718967305710e-01, 5.020389301172998309e-01, -6.687609208356652957e-01, -8.010382398744300136e-01, 3.286181123986433583e-01, -2.895598775918175605e-01,
                   1.766195678287743043e-01, -8.439209657449562307e-01, -5.585060066906111231e-01, -9.945484859184072768e-01, 9.875900827333250476e-03, 3.602415831568077653e-01, -7.186131598049803060e-01, 2.309956224097753363e-02, 7.169297365823896762e-01, 2.721447536623253782e-01, -2.113723814140890944e-01, -3.126397466019612548e-01, -6.956584416330782172e-01, -3.457626242440725584e-01, 6.061362634518534520e-01, 5.270782577735209085e-01, 9.300375316523779023e-01, 3.485232818604211413e-01, 6.610891260411504433e-01, 7.045726912282677112e-01, -5.988191328119747414e-01, 9.569629780720494416e-01, 3.388378385727519593e-01, -2.259940103506237197e-01, 3.635458363612946719e-01, 9.808537045708476398e-02, 2.599214025537590622e-01, 9.179560310054630801e-01, -3.704669737493961890e-01, -6.001428356786548957e-01, 3.397326046706015124e-01, 3.423049523603283184e-01]
    assert len(noise_input) is noise_dim, "noise_input length doesn't match GAN's noise_dim"
    for axis in range(noise_dim):
        steps = np.linspace(-1, 1, num=batch_size)
        noise_input_copy = np.array([noise_input, ] * batch_size)
        for i in range(len(steps)):
            noise_input_copy[i][axis] = steps[i]
        GAN.TestGAN(noise_input_copy, "TestRangeEachDimension-" + str(axis))


def TestRangeTwoDimensions():
    """Same as TestRangeEachDimension, but vary two dimensions each time. This will create ~64*64 = 4096, minus when the axes are the same."""
    batch_size = 100
    noise_dim = 64
    noise_input = [8.685266433488878501e-01, -7.861549933695841652e-01, -6.398974390956200242e-01, 3.079600179814192540e-01, 6.314542425683029592e-01, -3.832037598255819688e-01, -6.867196305925225008e-01, -6.538568355329785753e-01, 4.467145970933827925e-01, 1.074893006118486927e-01, -4.184085090469162882e-01, -9.668555709334403225e-01, -7.277596303315085891e-01, 7.232803008497166175e-01, -4.118989412568132558e-01, 7.643974076799808781e-01, 6.837788739535317628e-01, -3.154805591013811750e-01, 9.793365723614095852e-01, 1.570314484185821868e-01, 7.243522191672078225e-01, -9.833831630156151249e-01, -5.808857773322504414e-01, -6.119468805001466727e-01, 6.989781669440628953e-01, -4.548371455960735776e-01, 4.337164718967305710e-01, 5.020389301172998309e-01, -6.687609208356652957e-01, -8.010382398744300136e-01, 3.286181123986433583e-01, -2.895598775918175605e-01,
                   1.766195678287743043e-01, -8.439209657449562307e-01, -5.585060066906111231e-01, -9.945484859184072768e-01, 9.875900827333250476e-03, 3.602415831568077653e-01, -7.186131598049803060e-01, 2.309956224097753363e-02, 7.169297365823896762e-01, 2.721447536623253782e-01, -2.113723814140890944e-01, -3.126397466019612548e-01, -6.956584416330782172e-01, -3.457626242440725584e-01, 6.061362634518534520e-01, 5.270782577735209085e-01, 9.300375316523779023e-01, 3.485232818604211413e-01, 6.610891260411504433e-01, 7.045726912282677112e-01, -5.988191328119747414e-01, 9.569629780720494416e-01, 3.388378385727519593e-01, -2.259940103506237197e-01, 3.635458363612946719e-01, 9.808537045708476398e-02, 2.599214025537590622e-01, 9.179560310054630801e-01, -3.704669737493961890e-01, -6.001428356786548957e-01, 3.397326046706015124e-01, 3.423049523603283184e-01]
    assert len(noise_input) is noise_dim, "noise_input length doesn't match GAN's noise_dim"
    for axis1 in range(noise_dim):
        for axis2 in range(noise_dim):
            if axis1 == axis2:
                continue  # Only vary on two different dimensions
            steps1 = np.linspace(-1, 1, num=10)  # 10 steps each
            steps2 = np.linspace(-1, 1, num=10)  # So 10 * 10 = 100 faces each pair of axes
            noise_input_copy = np.array([noise_input, ] * batch_size)
            for i in range(len(steps1)):
                for j in range(len(steps2)):
                    noise_input_copy[i * 10 + j][axis1] = steps1[i]
                    noise_input_copy[i * 10 + j][axis2] = steps2[j]
            GAN.TestGAN(noise_input_copy, "TestRangeTwoDimensions-" + str(axis1) + "-" + str(axis2))


def MakeRandomFaces(outputDir, imgFilename, txtFilename):
    """Generate random faces"""
    batch_size = 25
    noise_dim = 64
    noise_input = MathLib.sample_z(batch_size, noise_dim)
    GAN.TestGANSingleImgOutput(noise_input, outputDir, imgFilename, txtFilename)


def MakeFaceByEncoding(encoding, outputDir, imgFilename, txtFilename):
    """Make a single face from encoding and put it in the outputDir, etc.
    - encoding: a list of floats."""
    GAN.TestGANSingleImgOutput([encoding], outputDir, imgFilename, txtFilename)


def MakeSimilarFaces(encoding, outputDir, imgFilename, txtFilename):
    """Make a grid of similar faces from encoding and put it in the outputDir, etc.
    - encoding: a list of floats."""
    batch_size = 25
    noise_input = utils.GenerateSimilarEncodings(encoding, batch_size)
    GAN.TestGANSingleImgOutput(noise_input, outputDir, imgFilename, txtFilename)

# def main():
#     # TestRangeEachDimension()
#     # TestRangeTwoDimensions()

# if __name__ == "__main__":
#     main()
