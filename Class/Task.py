class Task:
    def __init__(self, id, compution_number, ram_capacity):
        self.id = id  # parents=[], storage=0 ,core=None,deepcopy(TASKTERMINAL_dict)
        self.compution_number = compution_number
        self.ram_capacity = ram_capacity
        self.node = []
        # self.start_time = start_time
        # self.end_time = end_time

