# Options JSONs written by the released betas

Each `.json` here is byte for byte what a released beta wrote next to its exported
results. The JSON format changed at `7469418` ("harmonize options_jsons with single
source of truth"), between 1.0.0b2 and 1.0.0b3: before it the GUI and the CLI each had
their own writer. `tests/test_options.py` checks that today's
`magic_afm._options.load_options` reads each one and pins what it decodes. Later changes
to the options JSON (`fit_fix` as names, a schema version, a warning for the old `M`
default, volumes by name and the `trace` key) use them as their read-compat inputs.

| Fixture | Tag | Front end | Input file | Options |
| -- | -- | -- | -- | -- |
| `1.0.0b2-cli.json` | 1.0.0b2 | CLI | `PS_PCL_SAA/QNM/onekhz.000.pfc` | defaults |
| `1.0.0b2-cli-trace.json` | 1.0.0b2 | CLI | `PS_PCL_SAA/QNM/onekhz.000.pfc` | `--trace trace`, else defaults |
| `1.0.0b2-gui-pfc.json` | 1.0.0b2 | GUI | `PS_PCL_SAA/QNM/onekhz.000.pfc` | Retract selected, else defaults |
| `1.0.0b2-gui-ardf.json` | 1.0.0b2 | GUI | `PDMS_SAA/ffm/both0000.ARDF` | Retract selected, else defaults |
| `1.0.0b3-cli.json` | 1.0.0b3 | CLI | `PS_PCL_SAA/QNM/onekhz.000.pfc` | defaults |

Input paths are under `example.data/test_cases/` (stored `.xz`, unpacked to a temporary
directory). The `.pfc` has no trace attribute and has a sync distance, so its GUI JSON
has `"trace": null` and a float `sync_dist`. `both0000.ARDF` has trace and retrace
volumes and no sync distance, so its GUI JSON has `"trace": 1` and `"sync_dist": null`.
1.0.0b2 cannot read `.nid`, so all CLI runs use the `.pfc`, and the three CLI JSONs
share `k`, `defl_sens` and `sync_dist`.

## How they were made

Generated on 2026-10-15 by `make_options_fixtures.py` in this directory, which documents
the steps in its docstring:

```
git worktree add --detach WT_B2 1.0.0b2
git worktree add --detach WT_B3 1.0.0b3
pixi run -e test python tests/fixtures/options_json/make_options_fixtures.py \
    --b2 WT_B2 --b3 WT_B3 --out OUTDIR
cp OUTDIR/*.json tests/fixtures/options_json/
```

Each run is a child interpreter (`python -P`) with `PYTHONPATH` set to the worktree, and
asserts that it imported the tag's `magic_afm`.

- **CLI.** The tag's own click entry point (`magic_afm.cli._impl.main`) with
  `--disable-progress --stop-on-error --output-path DIR`, plus the options in the table.
  It fits the whole 32 x 32 file and writes `<stem>_options<Ret><Trace>.json`.
- **GUI, 1.0.0b2.** No window is opened. The writer is the tag's own
  `AsyncNavigationToolbar2Tk._aexport_calculations` (the "Export calculated images"
  button, `gui/_impl.py` lines 733-821, `write_options` at 779), run under `trio.run`
  with `tkinter.filedialog.asksaveasfilename` patched to return a path and no images to
  export. Its `_get_options` returns the tag's own `ForceVolumeTkDisplay.options`
  property (line 1530), as wired at line 2154. That property reads a stand-in display
  whose real tkinter variables (on a Tcl-only interpreter) hold the widget defaults from
  `ForceVolumeTkDisplay.__init__` (lines 1171, 1194, 1223-1250, 1260, 1294, 1314-1383)
  and `_add_parm` (line 1459): k, deflection sensitivity, sync distance and trace from
  the file's `AsyncFVFile.initial_parameters`; log10(M (Pa)) 6.0; radius 20.0; LJ scale
  2; the rest 0.0; "Estimate k sens." on; display z-d; fix radius; the other fix boxes
  from `FitFix.DEFAULTS`. The fit segment defaults to Skip, and the simulated user
  selected Retract, as one must before calculating property maps. The script asserts the
  resulting `fit_fix` equals the GUI's `fitfix_intvar` default (`FitFix.DEFAULTS`).
- **Shim.** 1.0.0b2's fit fails on numpy 2.5 (`float()` of a one-element array). The
  script applies the `magic_afm/calculation.py` hunks of `b53b6a0` ("bugfix new numpy
  scalar handling", released in 1.0.0b3) to the b2 worktree with `git apply -C1`. It
  touches the fit only; neither JSON writer is changed.

## Stack

Today's `test` environment on macOS arm64: Python 3.13.15, numpy 2.5.3, click 8.5.0,
cloup 3.1.0, attrs 26.1.0, trio 0.34.0, trio-parallel 1.3.0, matplotlib 3.11.1, soxr
1.1.0, numba 0.67.0. The 1.0.0b2 lock had click 8.2.1/8.3.0, cloup 3.0.8, attrs 25.3.0,
numpy 2.2.5/2.3.3 and Python 3.13.3/3.13.7 (no osx-arm64 entries). click 8.2 and later
return enum members from `click.Choice(Enum)`, so `fit_mode` and `trace` are handled the
same way as on the release stack; the JSON writers themselves use only `json` and the
enums.

## Encodings, as written

| Field | 1.0.0b2 CLI | 1.0.0b2 GUI | 1.0.0b3 |
| -- | -- | -- | -- |
| `fit_fix` | int (`123`) | int (`123`) | int (`123`) |
| `fit_mode` | name | name | name |
| `trace` | name (`"TRACE"`) or null | int (`1`) or null | name or null |
| `M` | `1e9` (default, meant as Pa) | `0.001` (GPa, from 10^6 Pa) | `1.0` (GPa) |
| `k_sens` | absent | present | present |
| key order | schema order | `ForceCurveOptions` order | schema order |

Today's `load_options` reads `1e9` as 1e9 GPa, with no warning.
