(use-modules (guix)
             ((guix licenses) #:prefix license:)
             (guix build-system pyproject)
             (guix build-system python)
             (guix build utils)
             (guix hg-download)
             (guix git-download)
             (guix utils)
             (common python-packages)
             (gnu packages)
             (gnu packages mpi)
             (gnu packages rust)
             (gnu packages rust-apps)
             (gnu packages statistics)
             (gnu packages bioinformatics)
             (gnu packages build-tools)
             (gnu packages pkg-config)
             (gnu packages python)
             (gnu packages python-build)
             (gnu packages python-xyz)
             (gnu packages python-science)
             (common python-fluidfft)
             (common python-fluiddyn)
             (common python-fluidsim-utils))

;; Define your commit hash here.
;; sha256 can be retrieved running:
;;    guix download [github-link] --commit=[commit-hash]
;; This will create a /gnu/store entry, and return sources hash in stdout.
(define fluidsim-src
  (origin
    (method git-fetch)
    (uri
      (git-reference
        (url "https://github.com/fluiddyn/fluidsim")
        (commit "f3d850a2bca41e750fc62a420341d8d7ccf800f8")))
    (sha256
      (base32 "0wfjkxv0d5hfw2qcjhfpw6czashxd8dj2fynkspikz3pg6s8ranc"))))

;; package python-fluidsim-core
(define-public python-fluidsim-core
  (package
    (name "python-fluidsim-core")
    (version "X.X.X")
    (source fluidsim-src)
    (build-system pyproject-build-system)
    (propagated-inputs 
      (list python-fluiddyn
            python-importlib-metadata))
    (arguments 
      '(#:phases
        (modify-phases %standard-phases
          ; On patche le HOME
          (add-before 'build 'patch-HOME-path
             (lambda _
               (chdir "lib")
               (setenv "HOME" (getenv "out"))))
          (add-before 'build 'patch-pyproject.toml
             (lambda _
               (invoke "sed" "-i" "s@license =.*@license = {text = 'CECILL-2.1'}@g" "pyproject.toml")))
          ;; On remove la phase de check et de sanity-check
          (delete 'check)
          (delete 'sanity-check))))
    (home-page "")
    (synopsis "Framework for studying fluid dynamics with simulations.")
    (description "Framework for studying fluid dynamics with simulations.")
    (license #f)))

;; package python-fluidsim
(define-public python-fluidsim
  (package
    (name "python-fluidsim")
    (version "X.X.X")
    (source fluidsim-src)
    (build-system pyproject-build-system)
    (propagated-inputs 
      (list python-fluidfft
            python-fluidsim-core
            python-h5netcdf
            python-h5py-mpi
            python-ipython
            python-matplotlib
            python-mpi4py
            python-pyfftw
            python-pymech
            python-rich
            python-scipy
            python-transonic
            python-xarray))
    (arguments 
      '(#:phases 
        (modify-phases %standard-phases
          ; On patche le HOME
          (add-before 'build 'patch-HOME-path
            (lambda _
              (setenv "HOME" (getenv "out"))))
          (add-before 'build 'patch-pyproject.toml
             (lambda _
               (invoke "sed" "-i" "s@license =.*@license = {text = 'CECILL-2.1'}@g" "pyproject.toml")))
          ;; On remove la phase de check et de sanity-check
          (delete 'check)
          (delete 'sanity-check))))
    (home-page "")
    (synopsis "Framework for studying fluid dynamics with simulations.")
    (description "Framework for studying fluid dynamics with simulations.")
    (license #f)))

;; Full environment manifest.
(concatenate-manifests
  (list
    (packages->manifest
      (list python-fluidsim))
    (specifications->manifest
      (list "python-fluidfft"
        "coreutils"
        ;"guix" ; don't think it's useful.
        "python-wrapper"
        "openmpi@4.1.6"
        "python-mpi4py"
        "python-h5py-mpi"
        "python-fluidfft-builder"
        "python-fluidfft-fftw"
        "python-fluidfft-fftwmpi"
        "python-fluidfft-mpi-with-fftw"
        "python-fluidfft-p3dfft"
        "python-fluidfft-pfft"
        "python-pytest"
        "python-pytest-allclose"
        "python-pytest-mock"
        ; build dependencies for editable build
        "meson-python"
        "python-pythran"
        ; convenient to be able to check
        "which"))))

