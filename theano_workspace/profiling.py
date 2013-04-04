from theano.compile.profiling  import ProfileStats

def add_profilers(ws):
    profs = {}
    for cu_name, cu in ws.compiled_updates.items():
        profs[cu_name] = ProfileStats(message=cu_name, flag_time_thunks=True)
        cu.profiler = profs[cu_name]
    return profs

