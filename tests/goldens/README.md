# Golden values

The goldens pin what the fitting code produces, so that any change to the numbers shows
up as a test failure and can be told apart from an intended change. There are two kinds,
each written by a generator script in this directory:

- `t2_goldens.json`, from `make_t2_goldens.py`: fitted values at the calculation layer
  in fit units, read by `tests/test_goldens.py`.
- `t3_goldens.json`, from `make_t3_goldens.py`: what the CLI writes to disk, read by
  `tests/test_cli.py`.

Each JSON file records the stack it was written on under `"stack"`.

## One shared synthetic set, one real-file set per platform

The files at the top of this directory hold values that every platform shares: the
synthetic curves (seeded, as the GUI's Demo builds them) fitted in each mode, and the
options JSON the CLI writes. Their tolerances span operating systems, and the comment
above `TOLERANCES` in `tests/test_goldens.py` gives the measured spread they cover.

Values fitted from real data files live in `<pixi platform>/` (`osx-arm64`, `linux-64`,
`win-64`) under the same file names. Fits of real curves depend on the CPU model, not
only the OS: OpenBLAS picks its kernels for the CPU it finds, numba tunes its code for
the host, and numpy dispatches to AVX-512 where it exists. Many real-file pixels are
ill-conditioned, so those small differences grow into differences larger than any
tolerance that would still catch a regression, and can move a pixel in or out of the NaN
mask. A set per platform, with the CPU-specific choices pinned, keeps each set
reproducible. `tests/conftest.py` `load_platform_goldens` picks the set for the running
platform.

One exception: in the Windows `test` environment (numba with numpy 2.5.4, the stack the
frozen Windows app ships) the real-file values are skipped, because even with every pin
applied they differ between the Zen 3 and Zen 4 runners GitHub hands out, while the
`test-nonumba` and `test-py311` environments agree across both. The `win-64` set was
made on a Zen 3 runner (EPYC 7763). The synthetic goldens still run there.

Treat the real-file sets as drift detectors: they say that something changed, not that
the new values are wrong. A deliberate numerics change is expected to trip them, and the
answer then is to regenerate them for every platform and say why in the commit.

## The pins

`pyproject.toml` sets these in the `test` feature's `activation.env`, per platform:

- `OPENBLAS_CORETYPE` makes OpenBLAS use one kernel set (`ARMV8` on macOS, `Haswell` on
  x86-64) whatever the CPU.
- `NUMBA_CPU_NAME = "haswell"` with an empty `NUMBA_CPU_FEATURES` makes numba compile
  for one fixed target; with the variable unset numba adds the host's own features.
- `NPY_DISABLE_CPU_FEATURES` turns off numpy's AVX-512 dispatch on x86-64.

They apply only inside the pixi test environments (`pixi run -e test ...`), and
`OPENBLAS_CORETYPE` works only because numpy there is the PyPI wheel, which bundles its
own OpenBLAS built for several CPUs. A numpy from conda-forge, or a plain `pip install`
and `pytest` outside pixi, runs unpinned and can fail the real-file goldens on a machine
whose CPU differs from the one that wrote them.

## Regenerating

Locally, for the platform you are on, from the repository root:

```
pixi run -e test python tests/goldens/make_t2_goldens.py --platform-only
pixi run -e test python tests/goldens/make_t3_goldens.py --platform-only
```

`--platform-only` leaves the shared files alone; drop it to rewrite them too. Use
`--out-dir DIR` to write somewhere else instead, for a comparison.

For the other platforms, CI's test job regenerates that platform's real-file set and
uploads it as an artifact named `goldens-<os>-<environment>-<attempt>`. It does so when
its pytest step fails, on a manual run of the Test workflow, and on a push whose head
commit message contains `[regen-goldens]`. Copy the `test` environment's files into
`tests/goldens/<pixi platform>/`.

The tests need the real data files, the `tests/magic-afm-test-data` submodule (its
`SHA256SUMS.txt` lists them); see the `real_data_file` fixture in `tests/conftest.py`.
Without them the real-file tests skip, unless `MAFM_REQUIRE_REAL_DATA=1` (set in CI)
makes them fail.

## When a golden test fails

The failure names the field and the case. To see how far a whole set moved, write a
fresh set to another directory and compare the two:

```
pixi run -e test python tests/goldens/make_t2_goldens.py --out-dir OTHER
pixi run -e test python tests/goldens/make_t2_goldens.py --compare \
    tests/goldens/osx-arm64/t2_goldens.json OTHER/osx-arm64/t2_goldens.json
```

`--compare A B` prints both stacks and then, per stored field, the largest absolute and
relative difference and whether the NaN pattern changed (`same` or `DIFFERENT`).
`make_t3_goldens.py` also reports changes in the sampled pixels and whether the options
are equal. A change confined to the last float32 digit (relative 1e-7) is rounding; a
changed NaN pattern or a large relative change in one platform's set alone usually means
the pins did not take effect there; a change in every set at once means the code or a
dependency changed the numerics.
