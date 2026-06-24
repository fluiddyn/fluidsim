import numpy as np
import h5py


def get_spectra_values_kh(data, strat=False):
    if strat:
        kh = data["kx"]
        Ekh_K = (data["spectra_E_kx"] + data["spectra_E_ky"]) / 2
        Ekh_A = (data["spectra_A_kx"] + data["spectra_A_ky"]) / 2
        Ekh_K[np.abs(Ekh_K) < 1e-15] = np.nan
        Ekh_A[np.abs(Ekh_A) < 1e-15] = np.nan
        valid_indices = np.where(~np.isnan(Ekh_K) & ~np.isnan(Ekh_A))[0]
        if len(valid_indices) == 0:
            return {
                "kh": np.array([]),
                "Ekh_K": np.array([]),
                "Ekh_A": np.array([]),
            }
        last_idx = valid_indices[-1] + 1
        return {
            "kh": kh[1:last_idx],
            "Ekh_K": Ekh_K[1:last_idx],
            "Ekh_A": Ekh_A[1:last_idx],
        }
    else:
        kh = data["kx"]
        Ekh_K = (data["spectra_E_kx"] + data["spectra_E_ky"]) / 2
        Ekh_K[np.abs(Ekh_K) < 1e-15] = np.nan
        valid_indices = np.where(~np.isnan(Ekh_K))[0]
        if len(valid_indices) == 0:
            return {"kh": np.array([]), "Ekh_K": np.array([])}
        last_idx = valid_indices[-1] + 1
        return {
            "kh": kh[1:last_idx],
            "Ekh_K": Ekh_K[1:last_idx],
        }


def get_spectra_values_kz(data, strat=False):
    if strat:
        kz = data["kz"]
        Ekz_K = data["spectra_E_kz"]
        Ekz_A = data["spectra_A_kz"]
        Ekz_K[np.abs(Ekz_K) < 1e-15] = np.nan
        Ekz_A[np.abs(Ekz_A) < 1e-15] = np.nan
        valid_indices = np.where(~np.isnan(Ekz_K) & ~np.isnan(Ekz_A))[0]
        if len(valid_indices) == 0:
            return {
                "kz": np.array([]),
                "Ekz_K": np.array([]),
                "Ekz_A": np.array([]),
            }
        last_idx = valid_indices[-1] + 1
        return {
            "kz": kz[1:last_idx],
            "Ekz_K": Ekz_K[1:last_idx],
            "Ekz_A": Ekz_A[1:last_idx],
        }
    else:
        kz = data["kz"]
        Ekz_K = data["spectra_E_kz"]
        Ekz_K[np.abs(Ekz_K) < 1e-15] = np.nan
        valid_indices = np.where(~np.isnan(Ekz_K))[0]
        if len(valid_indices) == 0:
            return {"kz": np.array([]), "Ekz_K": np.array([])}
        last_idx = valid_indices[-1] + 1
        return {
            "kz": kz[1:last_idx],
            "Ekz_K": Ekz_K[1:last_idx],
        }


def load_temp_average(keys=None, tmin=None, tmax=None, path_file=None):
    results = {}

    with h5py.File(path_file, "r") as file:
        times = file["times"][...]

        if keys is None:
            keys = [
                k
                for k in file.keys()
                if not any(
                    k.startswith(begin) for begin in ["r", "info_", "times"]
                )
            ]

        # Determine time range
        if tmax is None:
            tmax = times.max()
            imax_plot = np.argmax(times)
        else:
            imax_plot = np.argmin(abs(times - tmax))
            tmax = times[imax_plot]

        if tmin is None:
            tmin = times.min()
            imin_plot = np.argmin(times)
        else:
            imin_plot = np.argmin(abs(times - tmin))
            if imin_plot == imax_plot:
                if imin_plot == 0:
                    imax_plot += 1
                    tmax = times[imax_plot]
                else:
                    imin_plot -= 1
            tmin = times[imin_plot]

        # Load and average data
        for key in keys:
            results[key] = np.mean(file[key][imin_plot:imax_plot], axis=0)

    return results, tmin, tmax
