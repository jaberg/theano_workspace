import copy
import sys

import numpy as np

import theano
from theano.compile import deep_copy_op
from theano.compile.function_module import infer_reuse_pattern
from theano.compile.pfunc import rebuild_collect_shared
from theano.gof.vm import VM_Linker
from theano.printing import debugprint
from theano.sandbox.linalg.ops import Hint
from theano.sandbox.linalg.ops import is_hint_node


def optimizer_from_any(specifier):
    if isinstance(specifier, basestring):
        try:
            dct = theano.compile.mode.predefined_optimizers
            query = dct[specifier]
        except KeyError:
            raise ValueError('Optimizer %s not in %s' % (
                specifier, dct))
        return theano.compile.mode.optdb.query(query)
    else:
        # TODO probably not implemented error is more appropriate
        raise TypeError(specifier)


class UpdateFGraph(object):
    def __init__(self, updated_vars, givens=None):
        """
        updated_vars: sequence of (dst, expr) pairs
        vals_memo: dict Variable -> [value]

        """

        # -- unique_outputs is used here to ensure that there is some
        #    double-buffering going on, because actually dests and outputs can
        #    include some of the same variables (e.g. swap values)
        dests, outputs = zip(*updated_vars)
        unique_outputs = map(deep_copy_op, outputs)

        # -- partial graph clone to use givens
        stuff = rebuild_collect_shared(
            unique_outputs,
            inputs=[],
            replace=givens,
            rebuild_strict=True,
            copy_inputs_over=True)
        _inputs, unique_outputs_w_givens, other_stuff = stuff
        clone_equiv1, _update_d, _update_expr, _shared_inputs = other_stuff

        all_inputs = theano.gof.graph.inputs(unique_outputs_w_givens)

        # -- full graph clone to protect original graph
        clone_equiv = {}
        theano.gof.graph.clone_get_equiv(
            all_inputs,
            unique_outputs_w_givens,
            copy_inputs_and_orphans=True,
            memo=clone_equiv)
        for orig_var in clone_equiv1:
            clone_equiv[orig_var] = clone_equiv[clone_equiv1[orig_var]]
        self.cloned_inputs = [clone_equiv[var] for var in all_inputs]
        self.cloned_dests = [clone_equiv[var] for var in dests]
        self.cloned_outputs = [clone_equiv[var] for var in unique_outputs_w_givens]
        fgraph = theano.gof.fg.FunctionGraph(
            self.cloned_inputs,
            self.cloned_outputs)

        # -- load up fgraph with features necessary to maintain correctness:
        for node in fgraph.apply_nodes:
            if getattr(node.op, 'destroy_map', None):
                if not accept_inplace:
                    raise TypeError("Graph must not contain inplace operations", node, node.op)
                else:
                    fgraph.attach_feature(theano.gof.DestroyHandler())
                    break

        # We need to protect all immutable inputs from inplace operations.
        fgraph.attach_feature(
                theano.compile.function_module.Supervisor(invar
                    for invar in self.cloned_inputs
                    if not ((invar in self.cloned_dests) or
                            (hasattr(fgraph, 'destroyers') and
                                fgraph.destroyers(input)))))

        # If named nodes are replaced, keep the name
        for feature in theano.compile.function_module.std_fgraph.features:
            fgraph.attach_feature(feature())

        fgraph.attach_feature(theano.tensor.opt.ShapeFeature())

        # -- pre-install the shape information from the Hints created by
        #    e.g. SharedStorageWorkspace
        done = {}
        for node in fgraph.toposort():
            if is_hint_node(node):
                if node.inputs[0] in done: continue
                hints = dict(node.op.hints)
                if 'shape' in hints:
                    x = node.inputs[0]
                    assert x.ndim == len(hints['shape'])
                    if x in done:
                        assert done[x] == hints['shape']
                    else:
                        var_shape = tuple(
                            map(theano.tensor.as_tensor_variable,
                                hints['shape']))
                        fgraph.shape_feature.shape_of[node.inputs[0]] = var_shape
                        done[x] = hints['shape']

        self.updated_vars = updated_vars
        self.all_inputs = all_inputs
        self.outputs = outputs
        self.unique_outputs = unique_outputs
        self.clone_equiv = clone_equiv
        self.fgraph = fgraph


class CompiledUpdate(object):
    def __init__(self, ufgraph, vals_memo, profiler=None, **VM_Linker_kwargs):
        VM_Linker_kwargs.setdefault('use_cloop', True)
        VM_Linker_kwargs.setdefault('allow_gc', False)

        # -- create a VM to run the updates
        #    XXX CVM is necessary here until LoopGC implements updates
        linker = VM_Linker(**VM_Linker_kwargs)
        no_recycling = infer_reuse_pattern(ufgraph.fgraph, ufgraph.fgraph.outputs)
        linker.accept(ufgraph.fgraph, no_recycling=no_recycling)
        linker.accept_var_updates(dict(zip(
            ufgraph.cloned_dests,
            ufgraph.cloned_outputs)))

        input_storage = [vals_memo[i] if i in vals_memo else [i.data]
                for i in ufgraph.all_inputs]

        vm, input_containers, output_containers, thunks, order = linker.make_all(
            profiler=profiler,
            input_storage=input_storage,
            )

        self.ufgraph = ufgraph
        self.vals_memo = vals_memo
        self.input_storage = input_storage
        self.vm = vm
        self.input_containers = input_containers
        self.output_containers = output_containers
        self.thunks = thunks
        self.order = order

    def __call__(self):
        # if profiler then we need to update it (see function_module.py:641)
        return self.vm()


class Workspace(object):
    """

    This workspace is meant to be serializable, at least before it has been
    optimized.

    Recommended workflow for many repeated evaluations (pre-profile-driven
    optimization engine):
    1. build this type of workspace to define a function system
    2. use it to initialize a SharedStorageWorkspace, which will optimize the
       memory layout.
    3. call ws.optimize() on the SharedStorageWorkspace to optimize the
       computational graph for the optimized physical layout.
    4. run the optimized function system many times, it is the fastest.
    5. when it comes time to save, call ws.update(fast_ws) to bring the values
       back from the fast workspace to the original (slow) one, and save the
       slow one.
    """

    def __init__(self):
        self.vals_memo = {}
        self.compiled_updates = {}

    def __iter__(self):
        return self.vals_memo.keys()

    def __contains__(self, key):
        return key in self.vals_memo

    def __getitem__(self, key):
        return self.vals_memo[key][0]

    def __setitem__(self, key, val):
        filtered_val = key.type.filter(val, strict=False, allow_downcast=True)
        if key in self.vals_memo:
            self.vals_memo[key][0] = filtered_val
        else:
            self.vals_memo[key] = [filtered_val]

    def update(self, other):
        for key in other:
            self[key] = other[key]

    def add_method(self, name,
        inputs=None,
        outputs=None,
        updates=None,
        givens=None,
        optimizer=None,
        ):
        """Add a theano function as self.<name>

        Parameters
        ----------
        updates - a sequence of `(dest, expr)` pairs of theano variables
            When this function is called, it will update each workspace
            variable `dest` with the value computed for corresponding symbolic
            variable `expr`.

        """
        if inputs or outputs or givens:
            raise NotImplementedError()

        if not updates:
            raise NotImplementedError()

        ufgraph = UpdateFGraph(updates)
        if optimizer:
            ufgraph.optimize(optimizer)
        cu = CompiledUpdate(ufgraph, self.vals_memo)
        return self._add_compiled_update(name, cu)

    def _add_compiled_update(self, name, cu):
        self.compiled_updates[name] = cu
        setattr(self, name, cu)
        return cu

    def optimize(self, arg=None):
        """Convenient driver for various optimizations.

        The idea is to take time up front but reduce the average runtime of
        the methods (see `add_method()`) for future calls, assuming they
        continue to be used as they have been used so far.

        """
        # N.B. don't put too much effort into making this interface powerful
        # if a user wants more direct control over the optimizations, then
        # they should user a lower-level interface.
        self.optimize_storage()
        self.optimize_methods(arg)

    def optimize_storage(self):
        """Revise physical layout of values to enable faster methods.
        """

    def optimize_methods(self, specifier='fast_run'):
        """Recompile methods so they run faster, if possible.
        """
        optimizer = optimizer_from_any(specifier)
        for key, cu in self.compiled_updates.items():
            optimizer.apply(cu.ufgraph.fgraph)
            cu_opt = CompiledUpdate(cu.ufgraph, self.vals_memo)
            self.compiled_updates[key] = cu_opt


# XXX should this be a separate class, or just merged into the Workspace?
#     or maybe some kind of mix-in?
class SharedStorageWorkspace(Workspace):
    def __init__(self, ws):
        Workspace.__init__(self)

        self.views_memo = {}

        # set up some views
        self.s_fvector = theano.tensor.vector('fvector')
        vectors = [var for var in ws.vals_memo
                if var.type == self.s_fvector.type]
        if vectors:
            fvector = np.concatenate(
                    [ws.vals_memo[var][0] for var in vectors]).astype('float32')
            offset = 0
            for var in vectors:
                self.views_memo[var] = (
                        self.s_fvector,
                        offset,
                        len(ws.vals_memo[var][0]))
                offset += len(ws.vals_memo[var][0])
            self.vals_memo[self.s_fvector] = [fvector]

        #print self.views_memo

        # set up some normal values
        for var in ws.vals_memo:
            if var not in self.views_memo:
                self.vals_memo[var] = copy.deepcopy(ws.vals_memo[var])

        for fname, f in ws.compiled_updates.items():
            self.add_method(fname, updates=f.ufgraph.updated_vars)

    def __contains__(self, key):
        return key in self.vals_memo or key in self.views_memo

    def __getitem__(self, key):
        if key in self.views_memo:
            var, offset, n = self.views_memo[key]
            return self[var][offset: offset + n]
        else:
            return self.vals_memo[key][0]

    def __setitem__(self, key, val):
        filtered_val = key.type.filter(val, strict=False, allow_downcast=True)

        if key in self.views_memo:
            var, offset, n = self.views_memo[key]
            self.vals_memo[var][0][offset: offset + n] = filtered_val
        else:
            if key in self.vals_memo:
                self.vals_memo[key][0] = filtered_val
            else:
                self.vals_memo[key] = [filtered_val]

    def add_method(self, name,
        inputs=None,
        outputs=None,
        updates=None,
        givens=None,
        optimizer=None,
        ):
        noview_updates = dict() #XXX want ordered-dict here
        for dst, out in updates:
            if dst in self.views_memo:
                var, offset, n_elems = self.views_memo[dst]
                # -- build the shape into the graph
                #shp = self.vals_memo[var][0].shape
                # print 'shp', shp
                upvar = noview_updates.get(var, var)
                upvar = theano.tensor.set_subtensor(
                        upvar[offset: offset + n_elems],
                        out)
                noview_updates[var] = upvar
            else:
                if dst in noview_updates:
                    raise ValueError('duplicate destination', updated_vals)
                noview_updates[dst] = out

        givens = []
        for var in self.views_memo:
            svar, offset, n_elems = self.views_memo[var]
            shp = self.vals_memo[svar][0].shape
            svar = Hint(shape=shp)(svar)
            givens.append((var, svar[offset: offset + n_elems]))

        ufgraph = UpdateFGraph(noview_updates.items(), givens=givens)
        cu = CompiledUpdate(ufgraph, self.vals_memo)

        return self._add_compiled_update(name, cu)

