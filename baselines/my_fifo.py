def get_fifo_node_score(task, node):
    return -node.begin_idle_time

def get_fifo_task_score(task, node):
    return -task.arrive_time