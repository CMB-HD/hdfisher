from copy import deepcopy
import warnings


class FakeMPIComm:
    """A fake MPI communicator (very heavily inspired by `orphics.mpi`)"""
    
    def __init__(self):
        self.rank = 0
        self.size = 1

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size

    def Barrier(self):
        pass

    def barrier(self):
        pass




# try to import MPI, otherwise use fake MPI
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except:
    warnings.warn('Not using MPI')
    comm = FakeMPIComm()
rank = comm.Get_rank()
size = comm.Get_size()


def distribute(ntasks, size, rank):
    """Distribute the `ntasks` tasks among the `size` MPI ranks, 
    and return the task indices for this `rank`.

    Parameters
    ----------
    ntasks : int
        The total number of tasks to complete.
    size : int
        The total number of available MPI processes.
    rank : int
        The rank of the MPI process.

    Returns
    -------
    idxs : list of int
        A list of integer indices for the jobs assigned to this MPI `rank`.
    """
    # begin by giving each rank the same number of tasks, keeping track
    # of the leftovers:
    n, leftover = divmod(ntasks, size)
    # put the number of tasks for each MPI rank into a list, at the index
    # for that rank
    n_per_rank = [n] * size
    # distribute the leftovers, starting with the last rank
    for i in range(leftover):
        rank_idx = size - 1 - i
        n_per_rank[rank_idx] += 1
    # get the indices for this rank
    imin = sum(n_per_rank[:rank])
    imax = sum(n_per_rank[:rank+1])
    idxs = list(range(imin, imax))
    return idxs
