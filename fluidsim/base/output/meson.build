python_sources = [
  '__init__.py',
  'base.py',
  'cross_corr3d.py',
  'horiz_means.py',
  'increments.py',
  'phys_fields1d.py',
  'phys_fields2d.py',
  'phys_fields3d.py',
  'phys_fields.py',
  'print_stdout.py',
  'prob_dens_func.py',
  'spatial_means.py',
  'spatiotemporal_spectra.py',
  'spect_energy_budget.py',
  'spectra3d.py',
  'spectra_multidim.py',
  'spectra.py',
  'temporal_spectra.py',
  'time_signals_fft.py',
]

py.install_sources(
  python_sources,
  subdir: 'fluidsim/base/output'
)

run_command(
  ['transonic', '--meson', '--backend', backend, 'increments.py', 'spatiotemporal_spectra.py'],
  check: true
)

foreach be : backends
  subdir('__' + be + '__')
endforeach
