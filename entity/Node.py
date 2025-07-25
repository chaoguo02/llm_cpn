from core_function.update_time import computing_Task, computing_upload_time

class Node:
    def __init__(self, id, cpu_process, gpu_process, bandwidth, delay, cpu_capacity, ram_capacity, gpu_capacity, node_type, price):
        self.id = id
        self.node_type = node_type
        self.cpu_process = cpu_process #cpu单核计算能力
        self.gpu_process = gpu_process #gpu单核计算能力
        self.cpu_capacity = cpu_capacity #cpu核数
        self.ram_capacity = ram_capacity #ram容量
        self.gpu_capacity = gpu_capacity #gpu核数
        self.cpu_available = cpu_capacity
        self.ram_available = ram_capacity
        self.gpu_available = gpu_capacity
        self.bandwidth = bandwidth #带宽
        self.delay = delay
        self.waiting_queue = []  # 等待队列，为任务本身
        self.next_free_time = 0
        self.last_task_endtime = 0
        self.next_operation_time = 0
        self.begin_idle_time = 0 #node什么时候开始空闲
        self.completed_tasks_number=0 #已经处理过的节点数量
        self.completed_runningtime=0 #实际运行时间
        self.price = price

    # def __repr__(self):
        # return (f"<Node id={self.id}, type={self.node_type}, idle_time={self.begin_idle_time:.2f}, "
        #         f"cpu={self.cpu_capacity}, ram={self.ram_capacity}, gpu={self.gpu_capacity}, "
        #         f"price={self.price}>")

    def __repr__(self):
        return (
            f"<Node #{self.id} | Type: {self.node_type}\n"
            f"IdleTime: {self.begin_idle_time:.2f}s | Delay: {self.delay}ms | Bandwidth: {self.bandwidth}Mbps\n"
            f"CPU: {self.cpu_capacity} cores @ {self.cpu_process} | Available: {self.cpu_available}\n"
            f"RAM: {self.ram_capacity} units | Available: {self.ram_available}\n"
            f"GPU: {self.gpu_capacity} cores @ {self.gpu_process} | Available: {self.gpu_available}\n"
            f"Completed Tasks: {self.completed_tasks_number} | Runtime: {self.completed_runningtime:.2f}s\n"
            f"Price per sec: ${self.price}\n"
            f">"
        )

    def generate_heft_time(self,task):
        sum_time=self.begin_idle_time+computing_Task(task,self)+computing_upload_time(task,self)
        return sum_time
