theano_workspace
================

Workspace idea for theano. Workspaces are meant to provide a better
alternative to shared variables. Should help with serialization and general
interface usability, and enable more powerful optimizations.

```python

x = tensor.vector('x')
y = tensor.vector('y')

ws = Workspace()
ws[x] = [1, 2]
ws[y] = [3, 4]
ws.compile_update('f', [
    (x, 2 * x),
    (y, x + y)])

assert np.allclose([ws[x], ws[y]],[[1, 2], [3, 4]])

# compute stuff
ws.f()
assert np.allclose([ws[x], ws[y]],[[2, 4], [4, 6]])
```

