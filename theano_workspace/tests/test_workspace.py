import functools
import unittest

import numpy as np

import theano
from theano import tensor
from theano_workspace.workspace import Workspace, SharedStorageWorkspace

class StdMixins(object):
    def test_scaffolding(self):
        theano.printing.debugprint(self.foo[2].compiled_updates['f'].ufgraph.fgraph.outputs)

    def test_optimize(self):
        ws = self.foo[2]
        ws.optimize('fast_run')


class SimpleGraph(unittest.TestCase, StdMixins):
    def setUp(self):
        x = tensor.vector('x')
        y = tensor.vector('y')

        ws = Workspace()
        ws[x] = [1, 2]
        ws[y] = [3, 4]
        ws.add_method('f', updates=[(x, 2 * x), (y, x + y)])
        self.foo = x, y, ws

    def tearDown(self):
        x, y, ws = self.foo

        assert np.allclose([ws[x], ws[y]],[[1, 2], [3, 4]]), (ws[x], ws[y])
        ws.f()
        assert np.allclose([ws[x], ws[y]],[[2, 4], [4, 6]]), (ws[x], ws[y])
        ws.f()
        assert np.allclose([ws[x], ws[y]],[[4, 8], [6, 10]]), (ws[x], ws[y])


class SwapGraph(unittest.TestCase, StdMixins):
    def setUp(self):
        x = tensor.vector('x')
        y = tensor.vector('y')

        ws = Workspace()
        ws[x] = [1, 2]
        ws[y] = [3, 4]
        ws.add_method('f', updates=[(x, y), (y, x)])
        self.foo = x, y, ws

    def tearDown(self):
        x, y, ws = self.foo
        assert np.allclose([ws[x], ws[y]],[[1, 2], [3, 4]]), (ws[x], ws[y])
        ws.f()
        assert np.allclose([ws[x], ws[y]],[[3, 4], [1, 2]]), (ws[x], ws[y])
        ws.f()
        assert np.allclose([ws[x], ws[y]],[[1, 2], [3, 4]]), (ws[x], ws[y])
        ws.f()
        assert np.allclose([ws[x], ws[y]],[[3, 4], [1, 2]]), (ws[x], ws[y])
        ws.f()
        assert np.allclose([ws[x], ws[y]],[[1, 2], [3, 4]]), (ws[x], ws[y])


class MergeGraph2(unittest.TestCase, StdMixins):
    n_groups = 2
    n_items = 2
    def setUp(self):
        letters = 'xyzabcdefghijklmnopqrstuvw'
        symbols = [tensor.vector(a) for a in letters[:self.n_groups]]
        assert len(symbols) == self.n_groups

        ws = Workspace()
        for i, s in enumerate(symbols):
            ws[s] = range(i, i + self.n_items)
        f = ws.add_method('f', updates=[(s, 2 * s) for s in symbols])
        # XXX set up a new test with this update, because it seems
        # to break the optimizations
        #f = ws.add_method('f', updates=[(s, tensor.log(2 + tensor.cos(s))) for s in symbols])

        ws_shrd = SharedStorageWorkspace(ws)
        f_opt = ws_shrd.compiled_updates['f']
        self.foo = letters, symbols, ws, ws_shrd

    def tearDown(self):
        letters, symbols, ws, ws_shrd = self.foo

        for w in (ws, ws_shrd):
            # -- copy the variables
            ivals = [np.array(w[s]) for s in symbols]
            for i, s in enumerate(symbols):
                assert np.allclose(w[s], ivals[i])
            w.f()
            for i, s in enumerate(symbols):
                assert np.allclose(w[s], 2 * ivals[i]), (w[s], 2 * ivals[i])
            w.f()
            for i, s in enumerate(symbols):
                assert np.allclose(w[s], 4 * ivals[i]), (w[s], 4 * ivals[i])

    def test_storage_merged(self):
        ws, ws_shrd = self.foo[2:]
        assert len(ws.vals_memo) == self.n_groups, len(ws.vals_memo)
        assert len(ws_shrd.vals_memo) == 1, len(ws_shrd.vals_memo)

    def test_computation_merged(self):
        ws_shrd = self.foo[3]
        ws_shrd.optimize('fast_run')
        fgraph = ws_shrd.compiled_updates['f'].ufgraph.fgraph
        theano.printing.debugprint(fgraph.outputs)
        assert len(fgraph.toposort()) <= 2, len(fgraph.toposort())

    def test_timeit(self):
        import time
        ws, ws_shrd = self.foo[2:]
        ws.optimize('fast_run')
        ws_shrd.optimize('fast_run')
        fgraph = ws_shrd.compiled_updates['f'].ufgraph.fgraph
        theano.printing.debugprint(fgraph.outputs)
        def time_ws(w):
            times = []
            for i in range(100):
                t0 = time.time()
                w.f()
                t1 = time.time()
                times.append(t1 - t0)
            return times
        ws_times = time_ws(ws)
        ws_shrd_times = time_ws(ws_shrd)
        print 'n_groups=%s n_items=%s orig min %f' % (
                self.n_groups, self.n_items, min(ws_times))
        print 'n_groups=%s n_items=%s shrd min %f' % (
                self.n_groups, self.n_items, min(ws_shrd_times))


class MergeGraph5(MergeGraph2):
    n_groups = 5

class MergeGraph26_50(MergeGraph2):
    n_groups = 26
    n_items = 50

class MergeGraph26_1000(MergeGraph2):
    n_groups = 26
    n_items = 1000


