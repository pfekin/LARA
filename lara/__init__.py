"""LARA: Lightweight Additive Residual Adaptation.

Attach low-rank modules to a frozen language model's residual stream, train them
with any objective, and hold many of them resident on one base, routed per token.

    from lara import LARA, Bank

    lara = LARA(model, layers=6, rank=128)      # base frozen, ~2.4M trainable
    ...                                          # train with any trainer
    lara.save("behaviors/code", route_samples=texts)

    bank = Bank(model, tokenizer)
    bank.add("code", "behaviors/code")
    bank.add("polite", "behaviors/polite")
    bank.fit_router()
"""
from .core import LARA, LARAConfig, LARAModule, resolve_layers, decoder_layers
from .bank import Bank, Router

__version__ = "0.1.0"
__all__ = ["LARA", "LARAConfig", "LARAModule", "Bank", "Router",
           "resolve_layers", "decoder_layers"]
