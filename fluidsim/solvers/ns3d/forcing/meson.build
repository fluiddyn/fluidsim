python_sources = [
  '__init__.py',
  'milestone.py',
  'watu.py',
]

py.install_sources(
  python_sources,
  subdir: 'fluidsim/solvers/ns3d/forcing',
)

run_command(['transonic', '--meson', '--backend', backend, 'watu.py'], check: true)

foreach be : backends
  subdir('__' + be + '__')
endforeach
