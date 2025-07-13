from baselines.heuristic_score import decode1, decode2
from baselines.my_fifo import decode1_fifo, decode2_fifo
from baselines.my_greedy import decode1_greedy, decode2_greedy
from baselines.my_minmax import decode1_minmax,decode2_minmax
from baselines.my_minmin import decode1_minmin, decode2_minmin
from baselines.my_random import decode1_random, decode2_random
from baselines.my_sjf import decode1_sjf, decode2_sjf


def get_decode_functions(strategy):
    if strategy == 'gp':
        return decode1, decode2
    elif strategy == 'fifo':
        return decode1_fifo, decode2_fifo
    elif strategy == 'greedy':
        return decode1_greedy, decode2_greedy
    elif strategy == 'sjf':
        return decode1_sjf, decode2_sjf
    elif strategy == 'random':
        return decode1_random, decode2_random
    elif strategy == 'minmin':
        return decode1_minmin, decode2_minmin
    elif strategy == 'minmax':
        return decode1_minmax, decode2_minmax
    else:
        raise ValueError(f"Unknown strategy: {strategy}")