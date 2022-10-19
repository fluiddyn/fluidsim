import numpy as np


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
