import ConfigurationManager
import UtilityManager

configuration = ConfigurationManager.get_Config()
configuration.CAMERA_PATH = 'rtsp://root:pass@10.144.129.107/axis-media/media.amp'
configuration.FRAMERATE = 25
configuration.TEMP_VIDEO_PATH = 'algos/vid'
configuration.VIDEO_LENGTH = 5

UtilityManager.set_CUDA_Environment('0,1')
UtilityManager.ignore_Warning(True)
UtilityManager.check_Camera(configuration.CAMERA_PATH)

import link
link.set_configuration(configuration)
link.run(configuration)
