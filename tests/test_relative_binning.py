"""Tests for the relative-binning next-generation likelihood.

Relative binning approximates the ratio of a waveform to a fiducial waveform
as linear within each frequency bin, so near the fiducial point it must agree
with the exact ``GravitationalWaveTransientNextGeneration`` likelihood.
"""
import numpy as np
import pytest

import bilby
from bilby_xG.likelihood import (
    GravitationalWaveTransientNextGeneration,
    RelativeBinningGravitationalWaveTransientNextGeneration,
)
from bilby_xG.networks import InterferometerList
from bilby_xG.source import lal_binary_black_hole_relative_binning

lalsim = pytest.importorskip("lalsimulation")

DURATION = 4.0
SAMPLING_FREQUENCY = 1024.0
INJECTION = dict(
    mass_1=36.0, mass_2=29.0, a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0,
    phi_12=0.0, phi_jl=0.0, luminosity_distance=2000.0, theta_jn=0.4,
    psi=2.659, phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-1.2108,
    chi_1=0.0, chi_2=0.0,
)
WAVEFORM_ARGUMENTS = dict(
    waveform_approximant="IMRPhenomXP", reference_frequency=50.0,
    minimum_frequency=20.0,
)


def _waveform_generator(source_model):
    return bilby.gw.WaveformGenerator(
        duration=DURATION, sampling_frequency=SAMPLING_FREQUENCY,
        frequency_domain_source_model=source_model,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=WAVEFORM_ARGUMENTS.copy(),
    )


@pytest.fixture(scope="module")
def ifos():
    bilby.core.utils.random.seed(42)
    ifos = InterferometerList(["H1", "L1"])
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=SAMPLING_FREQUENCY, duration=DURATION,
        start_time=INJECTION["geocent_time"] - 2,
    )
    ifos.inject_signal(
        parameters=INJECTION,
        waveform_generator=_waveform_generator(bilby.gw.source.lal_binary_black_hole),
    )
    return ifos


@pytest.mark.parametrize("effects", [False, True])
def test_relative_binning_matches_exact(ifos, effects):
    flags = dict(
        earth_rotation_beam_patterns=effects, earth_rotation_time_delay=effects,
        finite_size=effects,
    )
    exact = GravitationalWaveTransientNextGeneration(
        interferometers=ifos,
        waveform_generator=_waveform_generator(bilby.gw.source.lal_binary_black_hole),
        **flags,
    )
    relbin = RelativeBinningGravitationalWaveTransientNextGeneration(
        interferometers=ifos,
        waveform_generator=_waveform_generator(lal_binary_black_hole_relative_binning),
        fiducial_parameters=dict(INJECTION), **flags,
    )
    for delta_mass_1 in [0.0, 0.3, -0.5]:
        parameters = dict(INJECTION, mass_1=INJECTION["mass_1"] + delta_mass_1)
        exact.parameters.update(parameters)
        relbin.parameters.update(parameters)
        logl_exact = exact.log_likelihood_ratio()
        logl_relbin = relbin.log_likelihood_ratio()
        assert np.isfinite(logl_relbin)
        assert logl_relbin == pytest.approx(logl_exact, abs=0.05)
