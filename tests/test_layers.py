import torch.nn as nn

from ntkmirror.layers import find_decoder_layers, parse_layers


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])


def test_find_decoder_layers():
    path, layers = find_decoder_layers(Tiny())
    assert path == "model.layers"
    assert len(layers) == 3


def test_parse_layers():
    assert parse_layers("all", 4) == [0, 1, 2, 3]
    assert parse_layers("last:2", 4) == [2, 3]
    assert parse_layers("0,-1", 4) == [0, 3]
