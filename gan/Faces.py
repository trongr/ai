import numpy as np
import DeconvGConvDVanillaGAN as GAN
import MathLib


def TestRangeEachDimension():
    """Generates 64 (noise_dim) images, each image contains 100 faces, each a variation from -1 to 1 of the same noise dimension."""
    batch_size = 100
    noise_input = [-0.117270297014, 0.134438872986, 0.14199347422, 0.476229903319, -0.61243681459, -0.184340664299, 0.00476305706842, 0.322285668428, 0.658127259621, -0.610947368402, -0.186660479461, -0.0358866743891, 0.833488707193, -0.852030146829, 0.139597235604, 0.622603574218, -0.849495877148, 0.439702176822, -0.618159443004, -0.0847780801927, -0.322158348938, -0.0828272649105, -0.397282086979, -0.0651804362691, -0.485027534351, 0.417959338337, 0.471490660839, 0.81071012994, -0.129992547979, 0.817216452082, 0.167688088943, 0.531721140134, -0.49823809883, -0.0964427942127, -0.939516260931, 0.522992550021, 0.0412466358619, 0.195790405804, -0.228978692583, 0.0638725704647, -0.313084505628, 0.812478573979, 0.509168347223, 0.281267636458, -0.372805721286, -0.107896490994, -0.476750528392, -0.024522976625, 0.516089012697, -0.206743212858, 0.130775650245, -0.441147815182, 0.244879645169, -0.592532255542, -0.122490132225, 0.182787455431, 0.201841894572, -1.03751767616, 0.954229958369, 0.175986279492, -0.746864495246, -0.241148894354, 0.104558866181, 0.0370786378922]
    noise_dim = 64
    assert len(noise_input) is noise_dim, "noise_input length doesn't match GAN's noise_dim"
    for axis in range(noise_dim):
        steps = np.linspace(-1, 1, num=batch_size)
        noise_input_copy = np.array([noise_input, ] * batch_size)
        for i in range(len(steps)):
            noise_input_copy[i][axis] = steps[i]
        GAN.TestGAN(noise_input_copy, axis)


def TestRandomFaces():
    batch_size = 100
    noise_dim = 64
    for it in range(1000):
        noise_input = MathLib.sample_z(batch_size, noise_dim)
        GAN.TestGAN(noise_input, it)


def main():
    # TestRangeEachDimension()
    TestRandomFaces()


if __name__ == "__main__":
    main()
