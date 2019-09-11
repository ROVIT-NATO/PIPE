import ConfigurationManager
import UtilityManager

configuration = ConfigurationManager.get_Config()
# configuration.CAMERA_PATH = 'rtsp://root:pass@10.144.129.107/axis-media/media.amp'
# configuration.CAMERA_PATH = 'rtsp://192.168.1.21:8554/live.ts'
configuration.CAMERA_PATH = 'video/shopping_center.mp4'
configuration.FRAMERATE = 25
configuration.TEMP_VIDEO_PATH = 'algos/vid/'
configuration.VIDEO_LENGTH = 5
configuration.GPU_ID = '0,1'
configuration.IS_ENABLE_WARNING = False
configuration.IS_RESIZE_INPUT_IMAGE = True

UtilityManager.set_CUDA_Environment(configuration.GPU_ID)
UtilityManager.enable_Warning(configuration.IS_ENABLE_WARNING)
UtilityManager.check_Camera(configuration.CAMERA_PATH)

import link

link.set_configuration(configuration)
link.run()
