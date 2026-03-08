import mc1._core
from mc1._core import DSP
from mc1.dag import *
from mc1.graphs import tone, drone

def perft(dag):
    if isinstance(dag, DAG):
        dag = bytes(dag)
    return mc1._core.perft(dag)
