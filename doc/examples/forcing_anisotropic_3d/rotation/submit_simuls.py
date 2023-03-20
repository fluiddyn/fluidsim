from util import Ro_target, submit

ns = [320, 640]

for n in ns:
    for NO_GEOSTROPHIC_MODES in [False, True]:
        for Ro in sorted(Ro_target):
            print("--------------------------------------------")
            submit(n,Ro,NO_GEOSTROPHIC_MODES)

