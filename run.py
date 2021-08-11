from System import pytorch_version
import platform


if __name__ == '__main__':
    print(">>>>  run.py begin")
    print(">>>>  Pytorch Version is %s" % pytorch_version())
    print(">>>>  System is %s" % platform.system().lower())