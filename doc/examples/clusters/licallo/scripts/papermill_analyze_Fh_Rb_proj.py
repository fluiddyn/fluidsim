"""
Script to execute the notebook notebook_analyse_Fr_Rb_proj.ipynb for several values of the parameters with papermill
"""

import papermill as pm

Fhs = [1e-2, 5e-3]
Rbs = [5, 10, 20, 40]
projs = ["None", "poloidal"]

for Fh in Fhs:
    for Rb in Rbs:
        for proj in projs:
            pm.execute_notebook(
                "notebook_analyze_Fh_Rb_proj.ipynb",
                f"analyze_Fh{Fh:.3e}_Rb{Rb:.3g}_proj{proj}.ipynb",
                parameters=dict(Fh=Fh, Rb=Rb, proj=proj),
            )
