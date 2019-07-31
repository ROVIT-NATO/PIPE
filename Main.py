import ProcessManager
import ConfigurationManager

settings = ConfigurationManager.get_Config()
settings.CAMERA_PATH = 'rtsp://root:pass@10.144.129.107/axis-media/media.amp'
settings.FRAMERATE = 25
settings.TEMP_VIDEO_PATH = 'algos/vid'
settings.VIDEO_LENGTH = 5

ProcessManager.run(settings)
