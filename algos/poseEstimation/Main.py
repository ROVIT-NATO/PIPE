import TestManager
import warnings
warnings.filterwarnings('ignore')

session = TestManager.init(InCheckPointPath='checkpoints/train/',
               vgg19_path='checkpoints/vgg/vgg_19.ckpt')
image = '/ocean/anish/Developer/RnD/poseEstimation/deep-high-resolution-net.pytorch/data/mpii/images/001030340.jpg'

TestManager.processFrame(image, session)

session[0].close()
