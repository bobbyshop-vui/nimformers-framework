"""ABI, context, and owned-tensor tests."""
import numpy as np
import pytest

from nimformers import Context, NimformersError, Tensor


def test_abi_version(ctx):
    assert ctx.abi_version == 2
    assert ctx.version.startswith("nimformers-capi/")


def test_backend_name(ctx):
    assert ctx.backend in {"metal", "vulkan", "cpu"}
    assert ctx.backend_id in {1, 2, 4}


def test_invalid_backend():
    with pytest.raises(ValueError):
        Context(backend="not-a-backend")


def test_invalid_library_path():
    with pytest.raises(NimformersError):
        Context(library="/nonexistent/libnimformers.dylib")


def test_sync(ctx):
    ctx.sync()


def test_tensor_roundtrip(ctx, rng):
    a = rng.standard_normal((3, 4)).astype(np.float32)
    t = ctx.tensor(a)
    assert isinstance(t, Tensor)
    assert t.shape == (3, 4)
    assert t.rank == 2
    assert t.numel == 12
    np.testing.assert_array_equal(t.numpy(), a)
    # numpy() is a copy by default
    copied = t.numpy()
    copied[0, 0] = 123.0
    assert t.numpy()[0, 0] != 123.0
    t.close()
    assert t.handle is None


def test_tensor_creation_helpers(ctx):
    z = ctx.zeros((2, 3), fill=1.5)
    np.testing.assert_array_equal(z.numpy(), np.full((2, 3), 1.5, np.float32))
    z.close()

    n = ctx.randn((4, 5), scale=0.01)
    assert n.shape == (4, 5)
    assert float(n.numpy().std()) < 0.05
    n.close()


def test_tensor_mutation(ctx, rng):
    a = rng.standard_normal((2, 6)).astype(np.float32)
    t = ctx.tensor(a)
    t.reshape(3, 4)
    assert t.shape == (3, 4)
    t.fill(0)
    assert np.all(t.numpy() == 0)
    b = rng.standard_normal((3, 4)).astype(np.float32)
    t.set_data(b)
    np.testing.assert_allclose(t.numpy(), b)
    with pytest.raises(ValueError):
        t.set_data(np.zeros(5, np.float32))
    with pytest.raises(NimformersError):
        t.reshape(5, 5)
    clone = t.clone()
    np.testing.assert_array_equal(clone.numpy(), t.numpy())
    clone.close()
    t.view(6, 2)
    assert t.shape == (6, 2)
    t.close()


def test_tensor_context_manager(ctx, rng):
    with ctx.tensor(rng.standard_normal(4).astype(np.float32)) as t:
        assert t.shape == (4,)
    assert t.handle is None


def test_low_level_error_surfaces(ctx):
    # A deliberately broken call must raise with the C API's message.
    with pytest.raises(ValueError):
        ctx.layernorm(np.ones((2, 3), np.float32),
                      np.ones(2, np.float32), np.zeros(2, np.float32))
