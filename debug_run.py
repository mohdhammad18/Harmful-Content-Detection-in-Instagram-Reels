import os
os.environ['QWEN_DEVICE'] = 'cpu'

from classifier import VideoCaptioningModel

try:
    m = VideoCaptioningModel()
    print('Model initialized successfully')
except Exception as e:
    import traceback
    traceback.print_exc()
    print('Exception:', e)
