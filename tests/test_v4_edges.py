import torch

from swccg_policy.v4_edges import degrees, sparse_edges


def fixture():
    tokens = torch.zeros((2, 5, 8), dtype=torch.int32)
    floats = torch.zeros((2, 5, 4))
    tokens[0, 2, [0, 5]] = torch.tensor([7, 1], dtype=torch.int32)
    tokens[0, 3, [0, 5, 7]] = torch.tensor([7, 2, 16], dtype=torch.int32)
    tokens[0, 4, [0, 5, 7]] = torch.tensor([7, 3, 8], dtype=torch.int32)
    tokens[1, 2, [0, 4, 5]] = torch.tensor([8, 3, 1], dtype=torch.int32)
    floats[1, 2, 0] = 2
    tokens[1, 3, [0, 5, 7]] = torch.tensor([7, 5, 64], dtype=torch.int32)
    return {"tok": tokens, "tokf": floats, "tmask": torch.ones((2, 5), dtype=torch.bool)}


def test_context_sources_alias_direction_and_binding():
    source, target, relation, binding, valid = sparse_edges(fixture())
    assert source.tolist() == [2, 3, 3, 5, 8]
    assert target.tolist() == [0, 1, 2, 6, 9]
    assert relation.tolist() == [13, 13, 15, 3, 16]
    assert binding.tolist() == [2, 3, 3, 7, 8]
    assert valid.all()
    counts = degrees(torch.zeros((10, 8)), source, target, valid)
    assert counts[:5].tolist() == [[0, 1], [0, 1], [1, 1], [2, 0], [0, 0]]


def test_packed_padding_carries_no_degree_or_binding():
    batch = fixture()
    natural = sparse_edges(batch)
    packed = sparse_edges({**batch, "edge_rows": torch.tensor([2, 3, 4, 7, 8, -1, -1])})
    for before, after in zip(natural, packed, strict=True):
        torch.testing.assert_close(before, after[:5])
    expected = degrees(torch.zeros((10, 8)), *natural[:2], natural[-1])
    actual = degrees(torch.zeros((10, 8)), *packed[:2], packed[-1])
    torch.testing.assert_close(expected, actual)


def test_empty_context_graph():
    batch = fixture()
    batch["tmask"].zero_()
    edges = sparse_edges(batch)
    assert all(len(values) == 0 for values in edges)
