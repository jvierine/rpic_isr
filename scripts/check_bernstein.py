"""Validate a completed Boris pilot without treating it as convergence evidence."""
import json
import sys
import h5py
import numpy as np

capture, spectrum = sys.argv[1:]
with h5py.File(capture) as f, h5py.File(spectrum) as g:
    config = json.loads(bytes(f['config_json'][:]))
    scales = json.loads(bytes(f['scales_json'][:]))
    assert config['bz_t'] > 0
    assert 'theory_psd' not in g
    for file in (f, g):
        def check(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.dtype.kind in 'fc':
                assert np.isfinite(obj[:]).all(), name
        file.visititems(check)
    raw = f['raw_signal'][:]
    assert np.array_equal(raw[:, 0], g['raw/0/real'][:])
    assert np.array_equal(raw[:, 1], g['raw/0/imag'][:])
    modes = f['electron_density_modes_1_to_8'][:]
    assert modes.shape == (len(raw), 8, 2)
    assert np.allclose(modes[:, config['bragg_mode']-1], raw, rtol=2e-5, atol=2e-5)
    assert f['positions_lambda_de'].shape == (4096,32768,2)
    d = g['diagnostics/0'][:].reshape(-1,9)
    excursion = np.max(abs(d[:,1]/d[0,1]-1))
    recording = d[d[:,0] >= config['burn_ms']*1e-3]
    print('energy excursion %:',100*excursion)
    print('record Te/Ti change %:',100*(recording[-1,5:7]/recording[0,5:7]-1))
    print('record seconds:',len(raw)*scales['dt_s'])
    print('frames:',len(f['positions_lambda_de']))
    assert excursion < .01, 'Pilot exceeds 1% energy excursion'
    assert abs(len(raw)*scales['dt_s']-config['duration_ms']*1e-3) < scales['dt_s']
    print('PASS finite data, voltage/modes, replay shape, duration, energy guard')
