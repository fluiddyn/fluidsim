(list (channel
        (name 'gricad-guix-packages)
        (url "https://gricad-gitlab.univ-grenoble-alpes.fr/bouttiep/gricad_guix_packages.git")
        (branch "master")
        (commit
          "222ba3b6ab33352a990a13562bc34663f452e006")) ;; 09/09/2025
      (channel
        (name 'guix)
        (url "https://git.guix.gnu.org/guix.git")
        (branch "master")
        (commit
          "ba17f399981f4f9f728902fe12896a750bf71856") ;; 09/09/2025
        (introduction
          (make-channel-introduction
            "9edb3f66fd807b096b48283debdcddccfea34bad"
            (openpgp-fingerprint
              "BBB0 2DDF 2CEA F6A8 0D1D  E643 A2A0 6DF2 A33A 54FA")))))
