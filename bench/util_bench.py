
from time import time
import pstats
import cProfile


def profile(sim):

    t0 = time()

    cProfile.runctx('sim.time_stepping.start()',
                    globals(), locals(), 'profile.pstats')
    t_end = time()
    if sim.oper.rank == 0:
        s = pstats.Stats('profile.pstats')
        # s.strip_dirs().sort_stats('time').print_stats(16)
        s.sort_stats('time').print_stats(12)

        times = print_analysis(s)

        print('\nelapsed time = {:.3f} s'.format(t_end - t0))

        print(
            '\nwith gprof2dot and graphviz (command dot):\n'
            'gprof2dot -f pstats profile.pstats | dot -Tpng -o profile.png')


def print_analysis(s):
    total_time = 0.
    times = {'fft2d': 0., 'fft_as': 0., 'pythran': 0., '.pyx': 0.}
    for key, value in s.stats.items():
        name = key[2]
        time = value[2]
        total_time += time
        for k in times.keys():
            if k in name or k in key[0]:
                if k == '.pyx':
                    if 'fft/Sources' in key[0]:
                        continue
                    if 'fft_as_arg' in key[2]:
                        continue

                if k == 'fft2d':

                    if 'util_pythran' in key[2] or \
                       'operators.py' in key[0] or \
                       'fft_as_arg' in key[2]:
                        continue

                    callers = value[4]

                    time = 0
                    for kcaller, vcaller in callers.items():
                        if 'fft_as_arg' not in kcaller[2] and\
                           'fft_as_arg' not in kcaller[0]:
                            time += vcaller[2]

                    # print(k, key)
                    # print(value[:100])
                    # print(time, '\n')

                if k == 'fft_as':
                    if '.pyx' in key[0]:
                        continue
                    # time = value[3]

                    # print(k, key)
                    # print(value[:100])
                    # print(time, '\n')

                times[k] += time

    print('Analysis (percentage of total time):')

    keys = list(times.keys())
    keys.sort(key=lambda key: times[key], reverse=True)

    for k in keys:
        t = times[k]
        print('time {:10s}: {:5.01f} % ({:4.02f} s)'.format(
            k, t/total_time*100, t))

    print('-' * 24 + '\n{:15s}  {:5.01f} %'.format(
        '', sum([t for t in times.values()])/total_time*100))

    return times
