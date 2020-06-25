from ctypes import *
dll=CDLL("./detector.dll")
print(dll.Loadmodel())