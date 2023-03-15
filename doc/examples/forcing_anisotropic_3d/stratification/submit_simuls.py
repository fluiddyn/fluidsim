from util import Fh_target, submit

ns = [320]

for n in ns:
    for NO_SHEAR_MODES in [False, True]:
        for Ro in sorted(Fh_target):
            print("--------------------------------------------")
            submit(n,Ro,NO_SHEAR_MODES)

