"""Symbolic strided views over flat row-major buffers.

A view maps a logical index ``i`` (one entry per dimension) to the flat element
offset ``offset + sum((i[d] // group[d]) * stride[d])``. ``group`` is normally 1;
values greater than one describe repeated rows, such as grouped-query K/V heads
expanded by slicing and concatenation or by ``expand`` followed by ``reshape``.
All rules are exact: an operation that cannot be expressed returns ``None``
and the caller materializes the value instead.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class View:
    offset: int
    dims: tuple  # ((size, stride, group), ...)

    @property
    def shape(self):
        return tuple(d[0] for d in self.dims)

    @property
    def size(self):
        return math.prod(self.shape)

    def at(self, index):
        return self.offset + sum((i // g) * s for i, (_, s, g) in zip(index, self.dims))

    @property
    def contiguous(self):
        """True when the view enumerates a dense row-major block."""
        expect = 1
        for size, stride, group in reversed(self.dims):
            if size == 1:
                continue
            if group != 1 or stride != expect:
                return False
            expect *= size
        return True


def row_major(shape):
    strides, step = [], 1
    for d in reversed(shape):
        strides.append(step)
        step *= d
    return tuple(reversed(strides))


def identity(shape, offset=0):
    return View(offset, tuple((d, s if d != 1 else 0, 1) for d, s in zip(shape, row_major(shape))))


def permute(view, dims):
    return View(view.offset, tuple(view.dims[d] for d in dims))


def slice_view(view, spec):
    offset, dims = view.offset, []
    for (size, stride, group), s in zip(view.dims, spec):
        if isinstance(s, int):
            offset += (s // group) * stride
            continue
        start, stop, step = s
        count = len(range(start, stop, step))
        if group == 1:
            offset += start * stride
            dims.append((count, stride * step if count > 1 else 0, 1))
        elif start % group == 0 and step == 1:
            offset += (start // group) * stride
            dims.append((count, stride, min(group, count) if count > 1 else 1))
        elif start % group == 0 and step % group == 0:
            offset += (start // group) * stride
            dims.append((count, stride * (step // group), 1))
        else:
            return None
    return View(offset, tuple(dims))


def broadcast_shape(a, b):
    out = []
    for i in range(1, max(len(a), len(b)) + 1):
        x = a[-i] if i <= len(a) else 1
        y = b[-i] if i <= len(b) else 1
        if x != y and x != 1 and y != 1:
            raise ValueError("shapes do not broadcast: %s, %s" % (a, b))
        out.append(max(x, y))
    return tuple(reversed(out))


def broadcast_to(view, shape):
    """Right-aligned broadcasting: size-1 dimensions get stride zero."""
    lead = len(shape) - len(view.dims)
    if lead < 0:
        return None
    dims = []
    for i, size in enumerate(shape):
        if i < lead:
            dims.append((size, 0, 1))
            continue
        have = view.dims[i - lead]
        if have[0] == size:
            dims.append(have)
        elif have[0] == 1:
            dims.append((size, 0, 1))
        else:
            return None
    return View(view.offset, tuple(dims))


def _split(dim, sizes):
    """Split one (size, stride, group) dimension into row-major ``sizes``."""
    size, stride, group = dim
    if math.prod(sizes) != size:
        return None
    out = []
    for n in reversed(sizes):
        if stride == 0 or group >= size:
            out.append((n, 0, 1))
        elif group == 1:
            out.append((n, stride, 1))
            stride *= n
        elif n % group == 0:
            out.append((n, stride, group))
            stride *= n // group
            group = 1
        elif group % n == 0:
            out.append((n, 0, 1))
            group //= n
        else:
            return None
        size //= n
    return tuple(reversed(out))


def _merge(dims):
    """Merge outer-to-inner dimensions into one, or return None."""
    dims = [d for d in dims if d[0] != 1]
    if not dims:
        return (1, 0, 1)
    size, stride, group = dims[-1]
    for outer in reversed(dims[:-1]):
        osize, ostride, ogroup = outer
        if ogroup != 1:
            return None
        if stride == 0 and group == 1:
            # An expanded inner dimension: rows repeat inside each outer row.
            if ostride == 0:
                size *= osize
                continue
            stride, group = ostride, size
            size *= osize
        elif ostride == stride * (size // group):
            size *= osize
        else:
            return None
    return (size, stride, group)


def reshape(view, shape):
    shape = tuple(shape)
    if math.prod(shape) != view.size:
        return None
    old = [d for d in view.dims if d[0] != 1]
    new = [n for n in shape if n != 1]
    out, i, j = [], 0, 0
    while i < len(old) or j < len(new):
        if i >= len(old) or j >= len(new):
            return None
        ochunk, nchunk = [old[i]], [new[j]]
        po, pn = old[i][0], new[j]
        i, j = i + 1, j + 1
        while po != pn:
            if po < pn:
                if i >= len(old):
                    return None
                ochunk.append(old[i])
                po *= old[i][0]
                i += 1
            else:
                if j >= len(new):
                    return None
                nchunk.append(new[j])
                pn *= new[j]
                j += 1
        merged = ochunk[0] if len(ochunk) == 1 else _merge(ochunk)
        if merged is None:
            return None
        if len(nchunk) == 1:
            out.append(merged)
        else:
            split = _split(merged, nchunk)
            if split is None:
                return None
            out.extend(split)
    dims, k = [], 0
    for n in shape:
        if n == 1:
            dims.append((1, 0, 1))
        else:
            dims.append(out[k])
            k += 1
    return View(view.offset, tuple(dims))


def cat(views, dim):
    """Concatenate views of one base along ``dim`` when the result stays strided."""
    first = views[0]
    rest = [d for k, d in enumerate(first.dims) if k != dim]
    for v in views[1:]:
        if [d for k, d in enumerate(v.dims) if k != dim] != rest:
            return None
    sizes = [v.dims[dim][0] for v in views]
    total = sum(sizes)
    if all(n == 1 for n in sizes):
        offsets = [v.offset - first.offset for v in views]
        run = 1
        while run < len(offsets) and offsets[run] == 0:
            run += 1
        if run == len(offsets):
            new = (total, 0, 1)
        else:
            stride = offsets[run]
            if len(offsets) % run or any(offsets[k] != (k // run) * stride for k in range(len(offsets))):
                return None
            new = (total, stride, run)
    else:
        stride = first.dims[dim][1]
        if any(v.dims[dim][2] != 1 or (v.dims[dim][0] > 1 and v.dims[dim][1] != stride) for v in views):
            return None
        position = 0
        for v, n in zip(views, sizes):
            if v.offset != first.offset + position * stride:
                return None
            position += n
        new = (total, stride, 1)
    dims = list(first.dims)
    dims[dim] = new
    return View(first.offset, tuple(dims))


def apply(node, graph, views):
    """Compose a view-producing node with the views of its inputs."""
    op = node.op
    if op == "reshape":
        return reshape(views[0], graph.values[node.output].shape)
    if op == "permute":
        return permute(views[0], node.attrs["dims"])
    if op == "slice":
        return slice_view(views[0], node.attrs["index"])
    if op == "expand":
        return broadcast_to(views[0], graph.values[node.output].shape)
    if op == "cat":
        return cat(views, node.attrs["dim"])
    return None


VIEW_NODE_OPS = ("reshape", "permute", "slice", "expand", "cat")


def resolve(graph, name, producers, stop=None):
    """Return (base, View) for ``name`` through view-only producers.

    ``stop`` lists value names that must be treated as bases.
    """
    cache = {}

    def walk(k):
        if k in cache:
            return cache[k]
        node = producers.get(k)
        if node is None or node.op not in VIEW_NODE_OPS or (stop and k in stop):
            result = (k, identity(graph.values[k].shape))
        else:
            parts = [walk(i) for i in node.inputs]
            bases = {b for b, _ in parts}
            view = None
            if len(bases) == 1 and all(v is not None for _, v in parts):
                view = apply(node, graph, [v for _, v in parts])
            result = (parts[0][0], view) if view is not None else (k, identity(graph.values[k].shape))
        cache[k] = result
        return result
    return walk(name)
