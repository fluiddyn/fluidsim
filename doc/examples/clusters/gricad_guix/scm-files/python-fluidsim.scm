(define-module (python-fluidsim)
  #:use-module (guix)
  #:use-module ((guix licenses) #:prefix license:)
  #:use-module (guix build-system pyproject)
  #:use-module (guix build-system python)
  #:use-module (guix build utils)
  #:use-module (guix hg-download)
  #:use-module (guix utils)
  #:use-module (common python-packages)
  #:use-module (gnu packages)
  #:use-module (gnu packages mpi)
  #:use-module (gnu packages rust)
  #:use-module (gnu packages rust-apps)
  #:use-module (gnu packages statistics)
  #:use-module (gnu packages bioinformatics)
  #:use-module (gnu packages build-tools)
  #:use-module (gnu packages pkg-config)
  #:use-module (gnu packages python)
  #:use-module (gnu packages python-build)
  #:use-module (gnu packages python-xyz)
  #:use-module (gnu packages python-science)
  #:use-module (common python-fluidfft)
  #:use-module (common python-fluiddyn)
  #:use-module (common python-fluidsim-utils))


;; package python-fluidsim-core
(define-public python-fluidsim-core
  (package
    (name "python-fluidsim-core")
    (version "X.X.X")
    (source
      (origin
        (method hg-fetch)
        (uri
          (hg-reference
            (url "https://foss.heptapod.net/fluiddyn/fluidsim")
            (changeset "dc37c60a95eb")))
        (sha256
          (base32 "1jpbqk0vvazmwzfwy4f3rcmjiz16f34aflb1v6nnw4pxhh29r1kq"))))
    (build-system pyproject-build-system)
    (propagated-inputs (list python-fluiddyn
                             python-importlib-metadata
                      ))
    (arguments '(#:phases (modify-phases %standard-phases
                   ; On patche le HOME
                   (add-before 'build 'patch-HOME-path
                      (lambda _
                        (chdir "lib")
                        (setenv "HOME" (getenv "out"))
                        ))
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
    (source
      (origin
        (method hg-fetch)
        (uri
          (hg-reference
            (url "https://foss.heptapod.net/fluiddyn/fluidsim")
            (changeset "dc37c60a95eb")))
        (sha256
          (base32 "1jpbqk0vvazmwzfwy4f3rcmjiz16f34aflb1v6nnw4pxhh29r1kq"))))
    (build-system pyproject-build-system)
    (propagated-inputs (list python-fluidfft
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
                             python-xarray
                      ))
    (arguments '(#:phases (modify-phases %standard-phases
                   ; On patche le HOME
                   (add-before 'build 'patch-HOME-path
                      (lambda _
                        (setenv "HOME" (getenv "out"))
                        ))
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

python-fluidsim
