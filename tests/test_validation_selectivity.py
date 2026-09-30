import numpy as np

from multi_training import combine_replica_rates, protocol_selectivity_from_rates


def test_protocol_selectivity_matches_direction_vector_definition():
    angles = np.arange(0, 360, 45)
    rates = np.zeros((8, 3), dtype=np.float32)
    rates[0, 0] = rates[4, 0] = 4.0
    rates[0, 1] = 4.0

    osi, dsi = protocol_selectivity_from_rates(rates, angles)

    np.testing.assert_allclose(osi[:2], [1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(dsi[:2], [0.0, 1.0], atol=1e-6)
    assert np.isnan(osi[2]) and np.isnan(dsi[2])


def test_replica_mean_rates_weight_all_simulated_trials():
    rates = [np.array([2.0, 4.0]), np.array([6.0, 8.0])]
    np.testing.assert_allclose(combine_replica_rates(rates, [32, 32]), [4.0, 6.0])
    np.testing.assert_allclose(combine_replica_rates(rates, [2, 1]), [10 / 3, 16 / 3])
