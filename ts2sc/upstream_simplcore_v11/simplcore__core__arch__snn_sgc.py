# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""Experimental signed-LIF spiking readout over the SGC mixer."""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch

from .sgc import LayerStreamState, SpectralGlobalConv, recurrent_dtype

__all__ = ["SpikingSpectralGlobalConv"]

_SPIKE_RESIDUAL_SCALE = 0.1


@torch.jit.script
def _lif_step(current: torch.Tensor, membrane: torch.Tensor,
              decay: float, threshold: float,
              slope: float) -> Tuple[torch.Tensor, torch.Tensor]:
    voltage = decay * membrane + (1.0 - decay) * current.to(membrane.dtype)
    hard = torch.where(
        voltage >= threshold, 1.0,
        torch.where(voltage <= -threshold, -1.0, 0.0))
    if torch.is_grad_enabled():
        surrogate = torch.tanh(slope * voltage)
        spikes = hard + surrogate - surrogate.detach()
    else:
        spikes = hard
    next_membrane = voltage - spikes.detach() * threshold
    return spikes.to(current.dtype), next_membrane


@torch.jit.script
def _lif_sequence(current: torch.Tensor, membrane: torch.Tensor,
                  decay: float, threshold: float,
                  slope: float) -> Tuple[torch.Tensor, torch.Tensor]:
    outputs = torch.jit.annotate(List[torch.Tensor], [])
    for t in range(current.size(1)):
        spikes, membrane = _lif_step(
            current[:, t], membrane, decay, threshold, slope)
        outputs.append(spikes)
    if current.size(1) == 0:
        return torch.empty_like(current), membrane
    return torch.stack(outputs, dim=1), membrane


class SpikingSpectralGlobalConv(SpectralGlobalConv):
    """SGC long-range mixer with a causal, signed LIF output stage.

    The SGC output remains the continuous residual path; a low-gain pulse
    correction is added on top. Its leaky membrane emits ternary events
    (-1, 0, +1) with a straight-through tanh surrogate gradient during training
    and an exact threshold in the forward pass. The membrane is constant-size
    state, so recurrent inference remains O(1) in context length.
    """

    def __init__(
        self, dim: int, num_bases: int = 64, decay_max: float = 8192.0,
        local_kernel: int = 4, content_gate: bool = True,
        out_gate: bool = False, evidence_gate: bool = False,
        uncertainty_gate: bool = False, norm: str = "rmsnorm",
        fft_dtype: Optional[torch.dtype] = torch.float32, eps: float = 1e-6,
        use_mixer_stream: bool = False, path_gate: bool = False,
        inst_nonlin: bool = False, init_mode: str = "logfreq",
        rho_param: str = "sigmoid", snn_decay: float = 0.9,
        snn_threshold: float = 0.1,
        snn_surrogate_slope: float = 5.0,
    ) -> None:
        if not math.isfinite(float(snn_decay)) \
                or not 0.0 <= float(snn_decay) < 1.0:
            raise ValueError(f"snn_decay must be in [0, 1), got {snn_decay}")
        if not math.isfinite(float(snn_threshold)) \
                or float(snn_threshold) <= 0.0:
            raise ValueError(
                f"snn_threshold must be positive, got {snn_threshold}")
        if not math.isfinite(float(snn_surrogate_slope)) \
                or float(snn_surrogate_slope) <= 0.0:
            raise ValueError(
                "snn_surrogate_slope must be positive, "
                f"got {snn_surrogate_slope}")
        super().__init__(
            dim=dim, num_bases=num_bases, decay_max=decay_max,
            local_kernel=local_kernel, content_gate=content_gate,
            out_gate=out_gate, evidence_gate=evidence_gate,
            uncertainty_gate=uncertainty_gate, norm=norm,
            fft_dtype=fft_dtype, eps=eps,
            use_mixer_stream=use_mixer_stream, path_gate=path_gate,
            inst_nonlin=inst_nonlin, init_mode=init_mode,
            rho_param=rho_param)
        self.snn_decay = float(snn_decay)
        self.snn_threshold = float(snn_threshold)
        self.snn_surrogate_slope = float(snn_surrogate_slope)

    def init_state(self, batch: int, device,
                   dtype) -> LayerStreamState:
        state = super().init_state(batch, device, dtype)
        state.spike_membrane = torch.zeros(
            batch, self.dim, device=device, dtype=recurrent_dtype(dtype))
        return state

    def _spike_step(self, current: torch.Tensor,
                    membrane: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if current.dim() != 2 or current.size(-1) != self.dim:
            raise ValueError(
                f"expected SNN current shaped (batch, {self.dim}), "
                f"got {tuple(current.shape)}")
        if membrane.shape != current.shape:
            raise ValueError(
                f"membrane shape {tuple(membrane.shape)} does not match "
                f"current shape {tuple(current.shape)}")
        membrane = membrane.to(recurrent_dtype(current.dtype))
        return _lif_step(
            current, membrane, self.snn_decay, self.snn_threshold,
            self.snn_surrogate_slope)

    def _spike_sequence(
        self, current: torch.Tensor, membrane: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if current.dim() != 3 or current.size(-1) != self.dim:
            raise ValueError(
                f"expected SGC current shaped (batch, time, {self.dim}), "
                f"got {tuple(current.shape)}")
        batch, length, _ = current.shape
        rd = recurrent_dtype(current.dtype)
        if membrane is None:
            membrane = torch.zeros(batch, self.dim, device=current.device,
                                   dtype=rd)
        else:
            membrane = membrane.to(device=current.device, dtype=rd)
        return _lif_sequence(
            current, membrane, self.snn_decay, self.snn_threshold,
            self.snn_surrogate_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        continuous = super().forward(x)
        spikes, _ = self._spike_sequence(continuous - x)
        return continuous + _SPIKE_RESIDUAL_SCALE * self.out_scale * spikes

    def forward_chunk(
        self, x: torch.Tensor, state: Optional[LayerStreamState] = None,
        *, update_state: bool = True,
    ) -> Tuple[torch.Tensor, Optional[LayerStreamState]]:
        continuous, next_state = super().forward_chunk(
            x, state, update_state=update_state)
        membrane = state.spike_membrane if state is not None else None
        spikes, final_membrane = self._spike_sequence(continuous - x, membrane)
        output = continuous + _SPIKE_RESIDUAL_SCALE * self.out_scale * spikes
        if state is not None and update_state:
            if next_state is None:
                raise RuntimeError("SGC chunk update returned no streaming state")
            next_state.spike_membrane = final_membrane.detach()
        return output, next_state

    @torch.no_grad()
    def step(self, x_t: torch.Tensor,
             state: LayerStreamState) -> Tuple[torch.Tensor, LayerStreamState]:
        continuous, state = super().step(x_t, state)
        membrane = state.spike_membrane
        if membrane is None:
            membrane = torch.zeros(
                continuous.shape[0], self.dim, device=continuous.device,
                dtype=recurrent_dtype(continuous.dtype))
        current = continuous - x_t
        if current.dim() == 3:
            if current.size(1) != 1:
                raise ValueError(
                    "SNN step expects exactly one token in the time dimension")
            current = current[:, 0]
        spikes, next_membrane = self._spike_step(current, membrane)
        if x_t.dim() == 3:
            spikes = spikes.unsqueeze(1)
        state.spike_membrane = next_membrane
        return (continuous + _SPIKE_RESIDUAL_SCALE * self.out_scale * spikes,
                state)

    def streaming_state_bytes(self, batch: int,
                              dtype=torch.float32) -> int:
        return (super().streaming_state_bytes(batch, dtype)
                + batch * self.dim
                * torch.empty(0, dtype=recurrent_dtype(dtype)).element_size())
