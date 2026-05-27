---
authors: ["Clovis Lambert", "Pierre Augier"]
abstract: |
  A executable notebook to analyse a simulation
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
exports:
  - format: typst
    template: lapreprint-typst
execute:
  depends_on_env: ["PATH_SIMUL_DIR"]

---

# Description of a simulation

```{code-cell}
import os

import numpy as np

import fluidsim
```

```{code-cell}
print(fluidsim)
```

```{code-cell} python
path_simul_dir = os.environ.get("PATH_SIMUL_DIR", None)
print(f"{path_simul_dir = }")
```
