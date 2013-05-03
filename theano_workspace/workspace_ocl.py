
from workspace import SimpleWorkspace, UpdateFGraph
import pyopencl as cl
import pyopencl.array

# -- map ops from e.g. theano.tensor
# -- to ones that work on OpenCL buffers
pre_alloc_reg = {}
gen_kernel_reg = {}

class CompiledUpdateOCL(object):
    def __init__(self, ufgraph, vals_memo, context, profiler=None):
        self.ufgraph = ufgraph
        self.vals_memo = vals_memo
        self.context = context
        self.queue = cl.CommandQueue(context)

        order = ufgraph.fgraph.toposort()
        storage_map = {}
        for var, varcell in vals_memo.items():
            storage_map[ufgraph.clone_equiv[var]] = varcell
        self.fn_args = []
        # -- pre-allocate all Variables
        # -- store command args for each node in a list
        # -- and generate OpenCL functions for each node
        for node in order:
            pre_alloc_reg[type(node.op)](self.queue, node, storage_map)
            fn, args = gen_kernel_reg[type(node.op)](self.queue, node, storage_map) 
            self.fn_args.append((fn, args))
        # XXX Copy the updates back to where they belong
        for var, expr in ufgraph.updated_vars:
            fn = cl.enqueue_copy
            dst = storage_map[ufgraph.clone_equiv[var]][0]
            src = storage_map[ufgraph.clone_equiv[expr]][0]
            args = (self.queue, dst.data, src.data,) # XXX add kwargs support to pass is_blocking=False
            self.fn_args.append((fn, args))
        self.storage_map = storage_map
            
    def __call__(self):
        for fn, args in self.fn_args:
            print fn, args
            fn(*args)
        self.queue.flush()

    def call_n_times(self, N):
        fn_args = self.fn_args
        for ii in xrange(N):
            for fn, args in fn_args:
                fn(*args)
        self.queue.flush()


class WorkspaceOCL(SimpleWorkspace):
    def __init__(self, ws, context=None):
        SimpleWorkspace.__init__(self)
        if context is None:
            context = cl.create_some_context()
        self.context = context
        queue = cl.CommandQueue(context)
        for var, valcell in ws.vals_memo.items():
            self.vals_memo[var] = [cl.array.to_device(queue, valcell[0])]
        queue.flush()

        for fname, f in ws.compiled_updates.items():
            self.add_method(fname, updates=f.ufgraph.updated_vars)


    def add_method(self, name,
        inputs=None,
        outputs=None,
        updates=None,
        givens=None,
        optimizer=None,
        ):
        if inputs or outputs or givens:
            raise NotImplementedError()

        if not updates:
            raise NotImplementedError()

        ufgraph = UpdateFGraph(updates)
        # XXX clone the 

        if optimizer:
            optimizer = optimizer_from_any(optimizer)
            optimizer.apply(ufgraph.fgraph)

        cu = CompiledUpdateOCL(ufgraph, self.vals_memo, self.context)
        return self._add_compiled_update(name, cu)

    def __getitem__(self, key):
        return self.vals_memo[key][0].get()

    def __setitem__(self, key, val):
        filtered_val = key.type.filter(val, strict=False, allow_downcast=True)
        if key in self.vals_memo:
            self.vals_memo[key][0].set(filtered_val)
        else:
            self.vals_memo[key] = [cl.array.to_device(filtered_val)]
