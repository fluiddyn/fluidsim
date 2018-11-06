from warnings import warn
import numpy as np


def load_txt(path_file, dict_results={}):
    warn(
        "Kept for legacy reasons to read old simulations, could be removed in the future",
        DeprecationWarning,
    )

    with open(path_file) as file_means:
        lines = file_means.readlines()

    lines_t = []
    lines_E = []
    lines_EK = []
    lines_epsK = []
    lines_epsA = []
    lines_epsCPE = []

    lines_epsK = []

    lines_PK = []
    lines_PA = []
    lines_etaskew = []
    lines_rotskew = []
    lines_Conv = []
    # Supplimentary dissipation
    lines_epsKsuppl = []

    for il, line in enumerate(lines):
        if line[0:6] == "time =":
            lines_t.append(line)
        if line[0:8] == "E      =":
            lines_E.append(line)
        if line[0:8] == "EK     =":
            lines_EK.append(line)
        if line[0:8] == "epsK   =":
            lines_epsK.append(line)
        if line[0:8] == "epsA   =":
            lines_epsA.append(line)
        if line[0:8] == "epsCPE =":
            lines_epsCPE.append(line)
        if line[0:8] == "PK1    =":
            lines_PK.append(line)
        if line[0:8] == "PA1    =":
            lines_PA.append(line)
        if line.startswith("eta skew ="):
            lines_etaskew.append(line)
        if line.startswith("rot skew ="):
            lines_rotskew.append(line)
        if line.startswith("Conv ="):
            lines_Conv.append(line)

        if line.startswith("epsKsup="):
            lines_epsKsuppl.append(line)

    nt = len(lines_t)
    # NOTE: unnecessary removal of last record
    #  if nt > 1:
    #      nt -= 1

    t = np.empty(nt)

    E = np.empty(nt)
    CPE = np.empty(nt)
    EK = np.empty(nt)
    EA = np.empty(nt)
    EKr = np.empty(nt)
    epsK = np.empty(nt)
    epsK_hypo = np.empty(nt)
    epsK_tot = np.empty(nt)
    epsA = np.empty(nt)
    epsA_hypo = np.empty(nt)
    epsA_tot = np.empty(nt)
    epsCPE = np.empty(nt)
    epsCPE_hypo = np.empty(nt)
    epsCPE_tot = np.empty(nt)

    if len(lines_PK) == len(lines_t):
        PK1 = np.empty(nt)
        PK2 = np.empty(nt)
        PK_tot = np.empty(nt)
        PA1 = np.empty(nt)
        PA2 = np.empty(nt)
        PA_tot = np.empty(nt)

    if len(lines_rotskew) == len(lines_t):
        skew_eta = np.empty(nt)
        kurt_eta = np.empty(nt)
        skew_rot = np.empty(nt)
        kurt_rot = np.empty(nt)

    if len(lines_Conv) == len(lines_t):
        Conv = np.empty(nt)
        c2eta1d = np.empty(nt)
        c2eta2d = np.empty(nt)
        c2eta3d = np.empty(nt)

    if len(lines_epsKsuppl) == len(lines_t):
        epsKsuppl = np.empty(nt)
        epsKsuppl_hypo = np.empty(nt)

    for il in range(nt):
        line = lines_t[il]
        words = line.split()
        t[il] = float(words[2])

        line = lines_E[il]
        words = line.split()
        E[il] = float(words[2])
        CPE[il] = float(words[6])

        line = lines_EK[il]
        words = line.split()
        EK[il] = float(words[2])
        EA[il] = float(words[6])
        EKr[il] = float(words[10])

        line = lines_epsK[il]
        words = line.split()
        epsK[il] = float(words[2])
        epsK_hypo[il] = float(words[6])
        epsK_tot[il] = float(words[10])

        line = lines_epsA[il]
        words = line.split()
        epsA[il] = float(words[2])
        epsA_hypo[il] = float(words[6])
        epsA_tot[il] = float(words[10])

        line = lines_epsCPE[il]
        words = line.split()
        epsCPE[il] = float(words[2])
        epsCPE_hypo[il] = float(words[6])
        epsCPE_tot[il] = float(words[10])

        if len(lines_PK) == len(lines_t):
            line = lines_PK[il]
            words = line.split()
            PK1[il] = float(words[2])
            PK2[il] = float(words[6])
            PK_tot[il] = float(words[10])

            line = lines_PA[il]
            words = line.split()
            PA1[il] = float(words[2])
            PA2[il] = float(words[6])
            PA_tot[il] = float(words[10])

        if len(lines_rotskew) == len(lines_t):
            line = lines_etaskew[il]
            words = line.split()
            skew_eta[il] = float(words[3])
            kurt_eta[il] = float(words[7])

            line = lines_rotskew[il]
            words = line.split()
            skew_rot[il] = float(words[3])
            kurt_rot[il] = float(words[7])

        if len(lines_Conv) == len(lines_t):
            line = lines_Conv[il]
            words = line.split()
            Conv[il] = float(words[2])
            c2eta1d[il] = float(words[6])
            c2eta2d[il] = float(words[10])
            c2eta3d[il] = float(words[14])

        if len(lines_epsKsuppl) == len(lines_t):
            line = lines_epsKsuppl[il]
            words = line.split()
            epsKsuppl[il] = float(words[1])
            epsKsuppl_hypo[il] = float(words[5])

    dict_results["t"] = t
    dict_results["E"] = E
    dict_results["CPE"] = CPE

    dict_results["EK"] = EK
    dict_results["EA"] = EA
    dict_results["EKr"] = EKr

    dict_results["epsK"] = epsK
    dict_results["epsK_hypo"] = epsK_hypo
    dict_results["epsK_tot"] = epsK_tot

    dict_results["epsA"] = epsA
    dict_results["epsA_hypo"] = epsA_hypo
    dict_results["epsA_tot"] = epsA_tot

    dict_results["epsCPE"] = epsCPE
    dict_results["epsCPE_hypo"] = epsCPE_hypo
    dict_results["epsCPE_tot"] = epsCPE_tot

    if len(lines_PK) == len(lines_t):
        dict_results["PK1"] = PK1
        dict_results["PK2"] = PK2
        dict_results["PK_tot"] = PK_tot
        dict_results["PA1"] = PA1
        dict_results["PA2"] = PA2
        dict_results["PA_tot"] = PA_tot

    if len(lines_rotskew) == len(lines_t):
        dict_results["skew_eta"] = skew_eta
        dict_results["kurt_eta"] = kurt_eta
        dict_results["skew_rot"] = skew_rot
        dict_results["kurt_rot"] = kurt_rot

    if len(lines_Conv) == len(lines_t):
        dict_results["Conv"] = Conv
        dict_results["c2eta1d"] = c2eta1d
        dict_results["c2eta2d"] = c2eta2d
        dict_results["c2eta3d"] = c2eta3d

    if len(lines_epsKsuppl) == len(lines_t):
        dict_results["epsKsuppl"] = epsKsuppl
        dict_results["epsKsuppl_hypo"] = epsKsuppl_hypo

    return dict_results
