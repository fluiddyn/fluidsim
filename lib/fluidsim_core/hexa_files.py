from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt

import pymech
from pymech.neksuite.field import read_header

from fluidsim_core.output.phys_fields import SetOfPhysFieldFilesBase


def get_edges_2d(var):
    ny, nx = var.shape
    edges = np.empty([ny + 1, nx + 1])
    for iy in [0, -1]:
        for ix in [0, -1]:
            edges[iy, ix] = var[iy, ix]

    for ix in [0, -1]:
        edges[1:-1, ix] = 0.5 * (var[:-1, ix] + var[1:, ix])

    for iy in [0, -1]:
        edges[iy, 1:-1] = 0.5 * (var[iy, :-1] + var[iy, 1:])

    edges[1:-1, 1:-1] = 0.25 * (
        var[:-1, :-1] + var[1:, 1:] + var[1:, :-1] + var[:-1, 1:]
    )
    return edges


class HexaField:
    def _init_from_arrays(self, arrays):
        self.arrays = [arr for arr in arrays]

    def _init_from_hexa_data(self, hexa_data, key, equation):
        if key.startswith("v"):
            name_attr = "vel"
            if key == "vx":
                index_var = 0
            elif key == "vy":
                index_var = 1
            elif key == "vz":
                index_var = 2
            else:
                raise ValueError
        elif key in "xyz":
            name_attr = "pos"
            if key == "x":
                index_var = 0
            elif key == "y":
                index_var = 1
            elif key == "z":
                index_var = 2
            else:
                raise ValueError

        elif key.startswith("temp"):
            name_attr = "temp"
            index_var = 0
        elif key.startswith("pres"):
            name_attr = "pres"
            index_var = 0
        else:
            raise NotImplementedError

        if equation is None:
            equation = "z=0"

        equation.replace(" ", "")

        self.arrays = []
        self.elements = []
        for elem in hexa_data.elem:
            arr_3d = getattr(elem, name_attr)[index_var]
            if equation.startswith("z="):
                z_target = float(equation[2:])
                z_3d = elem.pos[2]
                if not (z_3d.min() <= z_target <= z_3d.max()):
                    continue
                z_1d = z_3d[:, 0, 0]
                index_z = np.argmin(abs(z_1d - z_target))
                index_y = index_x = slice(None)
            else:
                raise NotImplementedError

            arr = arr_3d[index_z, index_y, index_x]

            self.arrays.append(arr)

            dict_elem = {"array": arr}

            if key in "xyz":
                dict_elem["edges"] = get_edges_2d(arr)
                dict_elem["lims"] = arr.min(), arr.max()

            self.elements.append(dict_elem)

        if key in "xy":
            self.lims = hexa_data.lims.pos[index_var]

        self.time = hexa_data.time

    def __init__(
        self, key, hexa_data=None, arrays=None, time=None, equation="z=0"
    ):
        self.key = key
        if hexa_data is None and arrays is not None:
            self._init_from_arrays(arrays)
        elif hexa_data is not None and arrays is None:
            self._init_from_hexa_data(hexa_data, key, equation)
        else:
            raise ValueError

        if time is not None:
            self.time = time

    def __mul__(self, arg):
        return self.__class__(
            self.key, arrays=[arg * arr for arr in self.arrays], time=self.time
        )

    def __add__(self, arg):

        return self.__class__(
            self.key,
            arrays=[arr0 + arr1 for arr0, arr1 in zip(arg.arrays, self.arrays)],
            time=(self.time + arg.time) / 2,
        )

    def min(self):
        result = np.inf
        for arr in self.arrays:
            min_elem = arr.min()
            if min_elem < result:
                result = min_elem
        return result

    def max(self):
        result = -np.inf
        for arr in self.arrays:
            max_elem = arr.max()
            if max_elem > result:
                result = max_elem
        return result


class SetOfPhysFieldFiles(SetOfPhysFieldFilesBase):
    def _get_data_from_time(self, time):
        index = self.times.tolist().index(time)
        return self._get_data_from_path(self.path_files[index])

    @lru_cache(maxsize=2)
    def _get_data_from_path(self, path):
        return pymech.open_dataset(path)

    def _get_hexadata_from_time(self, time):
        index = self.times.tolist().index(time)
        return self._get_hexadata_from_path(self.path_files[index])

    @lru_cache(maxsize=2)
    def _get_hexadata_from_path(self, path):
        return pymech.readnek(path)

    def _get_field_to_plot_from_file(self, path_file, key, equation):
        if equation is not None:
            raise NotImplementedError
        hexa_data = self._get_hexadata_from_path(path_file)
        hexa_field = HexaField(key, hexa_data)
        return hexa_field, float(hexa_data.time)

    def init_hexa_pcolormesh(
        self, ax, hexa_color, hexa_x, hexa_y, vmin=None, vmax=None
    ):

        images = []

        if vmax is None:
            vmax = hexa_color.max()

        if vmin is None:
            vmin = hexa_color.min()

        for (arr, elem_x, elem_y) in zip(
            hexa_color.arrays, hexa_x.elements, hexa_y.elements
        ):

            x_edges = elem_x["edges"]
            y_edges = elem_y["edges"]

            image = ax.pcolormesh(
                x_edges, y_edges, arr, shading="flat", vmin=vmin, vmax=vmax
            )

            images.append(image)

        fig = ax.figure

        cbar = fig.colorbar(images[0])

        def on_move(event):
            if event.inaxes is not None and event.inaxes == ax:
                x = event.xdata
                y = event.ydata

                elements_possibly_touched = []

                for (image, elem_x, elem_y) in zip(
                    images, hexa_x.elements, hexa_y.elements
                ):
                    xmin, xmax = elem_x["lims"]
                    ymin, ymax = elem_y["lims"]

                    x_2d = elem_x["array"]
                    y_2d = elem_y["array"]

                    if (xmin <= x <= xmax) and (ymin <= y <= ymax):
                        distance2_2d = (x_2d - x) ** 2 + (y_2d - y) ** 2
                        i1d = distance2_2d.argmin()
                        iy, ix = np.unravel_index(i1d, distance2_2d.shape)
                        distance2_min = distance2_2d[iy, ix]
                        color = image.get_array()[i1d]
                        elements_possibly_touched.append((distance2_min, color))

                if elements_possibly_touched:
                    elements_possibly_touched = sorted(
                        elements_possibly_touched, key=lambda el: el[0]
                    )
                    element_touched = elements_possibly_touched[0]
                    _, color = element_touched
                    fig.canvas.toolbar.set_message(
                        ax.format_coord(x, y) + f" {hexa_color.key} = {color:.3f}"
                    )

        fig.canvas.mpl_connect("motion_notify_event", on_move)

        return images, cbar

    def init_quiver_1st_step(self, hexa_x, hexa_y, percentage_dx_quiver=4.0):

        xmin = hexa_x.min()
        xmax = hexa_x.max()

        ymin = hexa_y.min()
        ymax = hexa_y.max()

        dx_quiver = percentage_dx_quiver / 100 * (xmax - xmin)
        nx_quiver = int((xmax - xmin) / dx_quiver)
        ny_quiver = int((ymax - ymin) / dx_quiver)

        x_approx_quiver = np.linspace(dx_quiver, xmax - dx_quiver, nx_quiver)
        y_approx_quiver = np.linspace(dx_quiver, ymax - dx_quiver, ny_quiver)

        indices_vectors_in_elems = []
        x_quiver = []
        y_quiver = []

        for x_2d, y_2d in zip(hexa_x.arrays, hexa_y.arrays):
            xmin = x_2d.min()
            xmax = x_2d.max()
            ymin = y_2d.min()
            ymax = y_2d.max()

            indices_vectors_in_elem = []

            for y_approx in y_approx_quiver:
                if y_approx < ymin:
                    continue
                if y_approx > ymax:
                    break
                for x_approx in x_approx_quiver:
                    if x_approx < xmin:
                        continue
                    if x_approx > xmax:
                        break

                    distance2_2d = (x_2d - x_approx) ** 2 + (y_2d - y_approx) ** 2

                    iy, ix = np.unravel_index(
                        distance2_2d.argmin(), distance2_2d.shape
                    )
                    indices_vectors_in_elem.append((iy, ix))

                    x_quiver.append(x_2d[iy, ix])
                    y_quiver.append(y_2d[iy, ix])

            indices_vectors_in_elems.append(indices_vectors_in_elem)

        return indices_vectors_in_elems, x_quiver, y_quiver

    def compute_vectors_quiver(self, hexa_vx, hexa_vy, indices_vectors_in_elems):
        vx_quiver = []
        vy_quiver = []
        vmax = -np.inf

        for indices_vectors_in_elem, arr_vx, arr_vy in zip(
            indices_vectors_in_elems, hexa_vx.arrays, hexa_vy.arrays
        ):

            vmax_elem = np.max(np.sqrt(arr_vx**2 + arr_vy**2))
            if vmax_elem > vmax:
                vmax = vmax_elem

            for iy, ix in indices_vectors_in_elem:
                vx_quiver.append(arr_vx[iy, ix])
                vy_quiver.append(arr_vy[iy, ix])

        return vx_quiver, vy_quiver, vmax

    def plot_hexa(
        self,
        key=None,
        time=None,
        equation="z=0",
        percentage_dx_quiver=4.0,
        vmin=None,
        vmax=None,
        normalize_vectors=True,
        quiver_kw={},
    ):
        if time is None:
            time = self.times[-1]
        else:
            time = self.times[abs(self.times - time).argmin()]
        hexa_data = self._get_hexadata_from_time(time)

        key_field = self.get_key_field_to_plot(key)
        hexa_field = HexaField(key_field, hexa_data, equation=equation)
        hexa_x = HexaField("x", hexa_data, equation=equation)
        hexa_y = HexaField("y", hexa_data, equation=equation)
        hexa_vx = HexaField("vx", hexa_data, equation=equation)
        hexa_vy = HexaField("vy", hexa_data, equation=equation)

        fig, ax = plt.subplots(layout="constrained")

        self.init_hexa_pcolormesh(
            ax, hexa_field, hexa_x, hexa_y, vmin=vmin, vmax=vmax
        )

        indices_vectors_in_elems, x_quiver, y_quiver = self.init_quiver_1st_step(
            hexa_x, hexa_y, percentage_dx_quiver=percentage_dx_quiver
        )
        vx_quiver, vy_quiver, vmax = self.compute_vectors_quiver(
            hexa_vx, hexa_vy, indices_vectors_in_elems
        )

        if normalize_vectors:
            vx_quiver /= vmax
            vy_quiver /= vmax

        ax.quiver(x_quiver, y_quiver, vx_quiver, vy_quiver, **quiver_kw)

        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        title = f"{key_field}, $t = {time:.3f}$"
        if vmax is not None:
            title += r", $|\vec{v}|_{max} = $" + f"{vmax:.3f}"
        ax.set_title(title)

    def time_from_path(self, path):
        header = self.get_header(path)
        return header.time

    def get_header(self, path=None):
        if path is None:
            path = self.path_files[0]
        with open(path, "rb") as file:
            return read_header(file)

    def _get_glob_pattern(self):
        session_id = self.output.sim.params.output.session_id
        case = self.output.name_solver
        return f"session_{session_id:02d}/{case}0.f?????"

    def get_vector_for_plot(self, time, equation=None):
        if equation is not None:
            raise NotImplementedError
        # temporary hack
        time = self.times[abs(self.times - time).argmin()]
        hexa_data = self._get_hexadata_from_time(time)
        vec_xaxis = HexaField(hexa_data=hexa_data, key="vx")
        vec_yaxis = HexaField(hexa_data=hexa_data, key="vy")
        return vec_xaxis, vec_yaxis

    def get_key_field_to_plot(self, key_prefered=None):
        if key_prefered is None:
            header = self.get_header()
            if "T" in header.variables:
                return "temperature"
            return "pressure"
        else:
            return key_prefered
