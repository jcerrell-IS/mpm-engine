# experiments/elastic

Stiffness, yield and nonlinear hyperelastic laws recovered from a gravity drop by
one convex solve. The blob is dropped in warp-mpm, the trajectory is dumped, and a
weak-form momentum residual linear in the material coefficients is solved by least
squares. The simulator is never differentiated.

Run:

```
.venv/bin/python -m experiments.elastic --help
.venv/bin/python -m experiments.elastic grid-gate      # the Step 0 acceptance gate
.venv/bin/python -m experiments.elastic all            # recover, gate, plastic, sample complexity, fe basis
```

## Results

| stage | number |
| --- | --- |
| `recover`, sphere, radial window | E 0.35 percent, nu 0.256 against 0.30, cond 4.4 |
| `recover`, cube, radial window | E 13.4 percent, nu 0.329, cond 16.4 |
| `grid-gate`, sphere, timeweak | E 0.102 percent, mu 0.212 percent, lam 0.966 percent |
| `grid-gate`, cube, timeweak | E 0.069 percent, mu 0.424 percent, lam 3.51 percent |
| `plastic-gate`, hard drop | yield identified to about 4 percent |
| `plastic-gate`, soft drop | yield refused, G still recovered |
| `hyperelastic`, Mooney, one gentle probe | C10 22.1, C01 36.0, Kbulk 8.4 percent |
| `hyperelastic`, Mooney, three hard probes | C10 15.9, C01 30.6, Kbulk 2.9 percent |
| `hyperelastic-fe`, Yeoh through K = 6 basis | W1(x) reconstruction relative L2 0.226 |
| `sample-complexity`, cube | Fisher eigenvalue ratio 1e4, bulk mode flat in N |

The cube row is why the grid-consistent route exists. The radial window vanishes
on a sphere surface but not on a flat face, so the floor contact traction leaks
into the residual and E comes out 13.4 percent high. The grid node rows of
`ident.weakform.elastic_grid` bring that to 0.069 percent, which is what let the
NCLaw comparison proceed.

## Consolidated from

| pre-consolidation script | now |
| --- | --- |
| `sim/elastic_drop.py` | `drop.py` stages recover, shape, errors |
| `sim/sample_complexity.py` | `drop.py` stage sample-complexity |
| `sim/elastic_grid_gate.py` | `grid_gate.py` |
| `sim/plastic_drop.py` | `plastic.py` |
| `sim/elastic_identify_sequential.py` | `sequential.py` elastic half |
| `sim/plastic_identify_sequential.py` | `sequential.py` plasticine half |
| `sim/hyperelastic.py` | `hyperelastic.py` |
| `sim/hyperelastic_fe.py` | `fe_basis.py` |

The drop-scene runner had two copies, and the radial interior-bump construction,
the validity filter and the row assembly loop had eight each, with drift between
them. `core.py` holds one version of each, and the two sequential scripts collapsed
into one estimator because they differ only in which stress basis fills the
columns. Code lines went 1411 to 1444, roughly flat: the stages CLI, the dump
dataclass and the type hints took the space the duplication freed. Total lines went
1842 to 2234, the increase being docstrings that now carry the results, the
artifact paths and a run command per module.

## Reproduction checks

Every number below was produced by the consolidated code on the dumps already in
the tree, before the originals were deleted, and compared to the recorded run.

| check | recorded | consolidated |
| --- | --- | --- |
| sphere radial, E_err | 0.0034888047331791405 | 0.0034888047331791405 |
| sphere radial, mu_err | 0.03885871961392828 | 0.03885871961392828 |
| sphere radial, nu | 0.25573903508066836 | 0.25573903508066836 |
| cube radial, E_err | 0.134340294155481 | 0.134340294155481 |
| cube radial, mu_err | 0.10918556658081369 | 0.10918556658081369 |
| grid gate, sphere timeweak E_err | 0.0010229504340479867 | 0.0010229504340479867 |
| grid gate, sphere instant E_err | 0.00627588194723272 | 0.00627588194723272 |
| grid gate, cube timeweak E_err | 0.000692258814248059 | 0.000692258814248059 |
| grid gate, cube instant E_err | 0.015033601343945935 | 0.015033601343945935 |
| grid gate, four acceptance flags | all true | all true |
| Mooney gentle, C10 / C01 / Kbulk err | 0.22100975 / 0.36008482 / 0.08380165 | identical |
| Mooney gentle, cond / rows / strain p99 | 713.4199870064517 / 5376 / 0.038577987 | identical |

The recorded sphere and cube radial numbers are the `radial_window` entries of
`video2sim/out/elastic_drop/grid_gate.json`; the grid rows are its `routes`
entries; the Mooney row is
`video2sim/out/hyperelastic/hyperelastic_recover.json`.

## One behavior change

`sim/elastic_identify_sequential.py` and `sim/plastic_identify_sequential.py`
filtered particles on `||F - I||`, which is not frame-objective: an element that
has rotated but barely strained gets dropped, and the SVD branch used to store F
differs between warp svd3 and numpy by a rotation that leaves the stress
identical. Every other script in the family had already moved to the
rotation-invariant `||log sigma(F)||`. `sequential.py` uses the rotation-invariant
filter, so its streamed numbers move:

| sphere stream, last frame | old `\|\|F - I\|\|` | new `\|\|log sigma\|\|` |
| --- | --- | --- |
| posterior E | 2.011713e5 (0.586 percent high) | 2.006977e5 (0.349 percent high) |
| posterior std of E | 507.82 | 490.64 |
| posterior nu | 0.252559 | 0.255739 |
| posterior std of nu | 6.94e-4 | 6.60e-4 |
| frame count, confident time | 449, 0.2020 s | 449, 0.2020 s |

The new value 2.006977e5 is the batch radial-window least-squares estimate to
seven digits, so the streamed posterior now lands on the batch solve rather than
0.24 percent away from it. The figures in `docs/writeup/figs` predate the change
and shift by less than a line width when regenerated.

## Artifacts

New artifacts go to `out/elastic/`. Reads search that directory first and then the
pre-consolidation locations, `out/elastic_drop`, `out/plastic_drop` and
`out/hyperelastic` under this repository and under the video2sim staging tree, so
a stage reuses the 676 MB cube dump instead of re-simulating it. Figures are
written to `out/elastic/` and copied into `video2sim/docs/writeup/figs`, which is
where the LaTeX includes them from.

Paths that older documents name still resolve, because nothing under the old
directories was moved or deleted. `docs/nclaw_comparison_plan.md` line 131 names
`out/elastic_drop/grid_gate.json`; that file is the recorded run, and the gate now
writes `out/elastic/grid_gate.json`.

## Limits

- The hyperelastic forward drops came from the JAX MPM engine (jmpm), which is no
  longer in this tree. `hyperelastic.run_drop` raises with the settings it used;
  the recovery stages read the existing dumps. Porting the Yeoh and Mooney forward
  materials to warp-mpm would make that stage self-contained.
- `grid_recover` needs the grid resolution the dump was produced at, because
  `run_drop` does not record it. The default 48 matches every dump in the tree.
- The soft plasticine drop reports a lower bound on the yield stress rather than a
  value. That is the intended output of the gate.
