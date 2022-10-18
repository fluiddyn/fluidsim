import numpy as np


def get_edges(var):
    edges = np.empty(var.size + 1)
    edges[0] = var[0]
    edges[-1] = var[-1]
    edges[1:-1] = 0.5 * (var[:-1] + var[1:])
    return edges


class HexaField:
    def _init_from_arrays(self, arrays):
        self.arrays = [arr for arr in arrays]

    def _init_from_hexa_data(self, hexa_data, key):
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
        else:
            raise NotImplementedError

        iz = 0
        self.arrays = []
        self.elements = []
        for elem in hexa_data.elem:
            arr = getattr(elem, name_attr)[index_var]
            self.arrays.append(arr)

            if key in "xy":

                if key == "x":
                    XX = arr[iz]
                    x = XX[0]
                    edges = get_edges(x)
                elif key == "y":
                    YY = arr[iz]
                    y = YY[:, 0]
                    edges = get_edges(y)

                self.elements.append(dict(edges=edges))

        if key in "xy":
            self.lims = hexa_data.lims.pos[index_var]

        self.time = hexa_data.time

    def __init__(self, key, hexa_data=None, arrays=None, time=None):
        self.key = key

        if hexa_data is None and arrays is not None:
            self._init_from_arrays(arrays)
        elif hexa_data is not None and arrays is None:
            self._init_from_hexa_data(hexa_data, key)
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
