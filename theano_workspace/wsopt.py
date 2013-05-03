import theano

from workspace import ViewWorkspace
from workspace import CompiledUpdate
from opt import optimizer_from_any


def optimize_methods(ws, query):
    """Recompile methods so they run faster, if possible.
    """
    optimizer = optimizer_from_any(query)
    for key, cu in ws.compiled_updates.items():
        optimizer.apply(cu.ufgraph.fgraph)
        cu_opt = CompiledUpdate(cu.ufgraph, ws.vals_memo)
        ws._add_compiled_update(key, cu_opt)
    return ws


def optimize(ws, query='fast_run'):
    """Convenient driver for various optimizations.

    The idea is to take time up front but reduce the average runtime of
    the methods (see `add_method()`) for future calls, assuming they
    continue to be used as they have been used so far.

    """
    # N.B. don't put too much effort into making this interface powerful
    # if a user wants more direct control over the optimizations, then
    # they should user a lower-level interface.
    ws = optimize_methods(ViewWorkspace(ws), query)
    return ws



