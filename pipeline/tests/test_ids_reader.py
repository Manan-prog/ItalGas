"""Format reader: the .index radar clock (signed 32-bit microseconds) is unwrapped across its 2^32 us rollover."""
import numpy as np

from ids_reader import INDEX_DTYPE, INDEX_HEADER_BASE_BYTES, read_index


def test_index_clock_unwrapped(tmp_path):
    n_ch = 10; t_true = np.arange(2147.0e6, 2149.0e6, 0.25e6).astype(np.int64)          # crosses 2^31 us
    rec = np.zeros(len(t_true), INDEX_DTYPE); rec["scan"] = np.arange(len(t_true)); rec["odo_m"] = np.arange(len(t_true)) * 0.045
    rec["t_us"] = ((t_true + (1 << 31)) % (1 << 32) - (1 << 31)).astype(np.int32)      # what the instrument stores
    f = tmp_path / "x.index"; f.write_bytes(b"\0" * (INDEX_HEADER_BASE_BYTES + 4 * n_ch) + rec.tobytes())
    ix = read_index(f, n_ch)
    assert np.all(np.diff(ix["t_us"]) == 250000)
    assert ix["t_us"][0] == t_true[0] and ix["t_us"][-1] - ix["t_us"][0] == t_true[-1] - t_true[0]
    assert np.allclose(ix["odo_m"], rec["odo_m"])


def test_index_without_wrap_unchanged(tmp_path):
    rec = np.zeros(5, INDEX_DTYPE); rec["t_us"] = [10, 20, 30, 40, 50]
    f = tmp_path / "y.index"; f.write_bytes(b"\0" * (INDEX_HEADER_BASE_BYTES + 4 * 19) + rec.tobytes())
    assert read_index(f, 19)["t_us"].tolist() == [10, 20, 30, 40, 50]
