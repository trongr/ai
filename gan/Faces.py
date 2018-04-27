import numpy as np
import DeconvGConvDVanillaGAN as GAN

batch_size = 100
# noise_input = [0.381935094072, 0.993223977256, -0.917976787045, -0.738318301788, 0.983937322845, 0.568823153852, 0.799077259729, 0.308355393499, -0.273507900704, 0.99261561491, -0.301363411266, -0.378580213994, 0.553637597003, 0.655594412744, -0.10589809701, -0.445392174697, 0.127209381136, 0.279231318717, -0.767187890619, 0.517912382384, -0.982118335871, 0.68021891118, 0.550204859767, -0.405170726768, -0.209793820516, 0.32421283446, 0.655606459433, 0.455121130887, 0.444072844148, -0.723755365999, -0.876505903235, 0.154755187644,
#                0.103318084893, -0.813127115237, 0.882995313363, -0.195894568946, -0.761815096228, 0.991532449875, 0.0581586407051, 0.240098388243, 0.905119550972, -0.593938262809, -0.0490899453885, -0.505825671087, -0.86150670744, -0.969691452214, 0.265612969146, -0.67898421121, -0.849759991117, 0.396833010409, -0.936391424904, -0.573455737039, -0.667525119719, -0.278111298132, -0.155129912759, 0.979012054522, -0.31859680795, 0.542003448302, -0.984675780767, 0.223453406233, -0.825112411321, 0.735301118248, -0.587375611399, -0.100033493553]
noise_input = [-0.117270297014,0.134438872986,0.14199347422,0.476229903319,-0.61243681459,-0.184340664299,0.00476305706842,0.322285668428,0.658127259621,-0.610947368402,-0.186660479461,-0.0358866743891,0.833488707193,-0.852030146829,0.139597235604,0.622603574218,-0.849495877148,0.439702176822,-0.618159443004,-0.0847780801927,-0.322158348938,-0.0828272649105,-0.397282086979,-0.0651804362691,-0.485027534351,0.417959338337,0.471490660839,0.81071012994,-0.129992547979,0.817216452082,0.167688088943,0.531721140134,-0.49823809883,-0.0964427942127,-0.939516260931,0.522992550021,0.0412466358619,0.195790405804,-0.228978692583,0.0638725704647,-0.313084505628,0.812478573979,0.509168347223,0.281267636458,-0.372805721286,-0.107896490994,-0.476750528392,-0.024522976625,0.516089012697,-0.206743212858,0.130775650245,-0.441147815182,0.244879645169,-0.592532255542,-0.122490132225,0.182787455431,0.201841894572,-1.03751767616,0.954229958369,0.175986279492,-0.746864495246,-0.241148894354,0.104558866181,0.0370786378922]
noise_dim = 64
assert len(noise_input) is noise_dim, "noise_input length doesn't match GAN's noise_dim"
# for axis in range(noise_dim):
for axis in range(1):
    steps = np.linspace(-1, 1, num=batch_size)
    noise_input_copy = np.array([noise_input, ] * batch_size)
    for i in range(len(steps)):
        noise_input_copy[i][axis] = steps[i]
    GAN.test(noise_input_copy, str(axis))
