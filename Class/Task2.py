"""
CPU 单位为GHz(任务所需计算量)
RAM 单位为GB

"""

# 任务类
class Task:
    def __init__(self, id,runtime,cpu_require, ram_require, gpu_require):
        self.uploadSize = 300 #文件大小
        self.outputFileSize = 300
        self.uploadtime = 0  # 上传时间
        self.outputtime = 0  # 下载时间
        self.transtime = 0  # 传输时间

        self.runtime = runtime
        self.exetime = 0
        self.starttime = None
        self.endtime = None
        self.given_starttime = None #开始时间
        self.given_endtime = None #结束时间
        self.id = id
        self.length = None #任务长度
        self.cpu_require = cpu_require #要求核数
        self.ram_require = ram_require #GB
        self.gpu_require = gpu_require #百分比

        self.cpu_allocated = cpu_require  # 要求核数
        self.ram_allocated = ram_require  # GB
        self.gpu_allocated = gpu_require  # 百分比
        self.node = None

