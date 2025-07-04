from fluiddyn.clusters.gricad import (
    Dahu as _Dahu,
    Dahu16_6130 as _Dahu16_6130,
    Dahu16_6244 as _Dahu16_6244,
    Dahu24_6126 as _Dahu24_6126,
    Dahu32_5218 as _Dahu32_5218,
    Dahu32_6130 as _Dahu32_6130,
    DahuDevel as _DahuDevel,
)


class Dahu(_Dahu):
    commands_setting_env = [
        "GUIX_PROFILE=$HOME/guix-profile-fluidsim",
        "source $GUIX_PROFILE/etc/profile",
        "export OMPI_MCA_plm_rsh_agent=/bettik/legi/oar-envsh",
        "export OMPI_MCA_btl_openib_allow_ib=true",
        "export OMPI_MCA_pml=cm",
        "export OMPI_MCA_mtl=psm2",
    ]

    def submit_command(self, command, *args, **kwargs):
        command = "--prefix $GUIX_PROFILE " + command
        return super().submit_command(command, *args, **kwargs)


class Dahu16_6130(Dahu, _Dahu16_6130):
    pass


class Dahu16_6244(Dahu, _Dahu16_6244):
    pass


class Dahu24_6126(Dahu, _Dahu24_6126):
    pass


class Dahu32_5218(Dahu, _Dahu32_5218):
    pass


class Dahu32_6130(Dahu, _Dahu32_6130):
    pass


class DahuDevel(Dahu, _DahuDevel):
    pass
