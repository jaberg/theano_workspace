from theano.compile.profiling  import ProfileStats

def add_profilers(ws, time_thunks=True, atexit_print=False):
    profs = {}
    for cu_name, cu in ws.compiled_updates.items():
        profs[cu_name] = ProfileStats(
            message=cu_name,
            flag_time_thunks=time_thunks,
            atexit_print=atexit_print,
            )
        cu.profiler = profs[cu_name]
    return profs

