# rtdetr_detection_colab — fleet-sweep fixes (2026-10-05)

Targeted fix of the 2026-10-05 fleet sweep findings. There is no full Notebook Review Framework v1 report for this
notebook; each flag was first confirmed in the cell source on `main` (`7508f6c`). All changes are made in the
generator (`tools/build_notebook.py`, `tools/notebook_template*.py`) and the notebook is regenerated. STATUS and the
release labels are unchanged. **Readiness: Verification pending** (hosted Run all not yet done).

## Findings and fixes

| ID | Status | Change | Cells / files touched | Evidence |
|---|---|---|---|---|
| SWP-R | Fixed — hosted confirmation pending | Confirmed: Section 1 ran `pip install` into the kernel and raised "Restart the runtime" on stale modules. Generator upgraded to `build_notebook.py/2.2` (the fleet isolated runtime): one kernel cell downloads the pinned `uv` 0.12.15 wheel (size + SHA-256), builds a managed CPython 3.12.12 environment from the new hash lock `tutorials/requirements-colab.lock.txt` (`--require-hashes --only-binary :all:`), and routes every later cell to one persistent worker. The environment folder is keyed on the lock digest and reused by a re-run or a second Run all; re-running Section 1 keeps the live worker and its variables; the worker gets `MPLBACKEND=Agg` and no `PYTHONPATH`/`PYTHONHOME`/`PYTHONSTARTUP`. | Section 1 (kernel cell + "Record the runtime"); `tools/build_notebook.py`; `tutorials/requirements-colab.lock.txt`; `tools/validate_release_assets.py` (install markers, bootstrap check, kernel cell excluded from the library-use scan); `docs/release-verification.md` (the line describing that check) | `test_swp_r_no_pip_install_or_restart_in_any_cell`, `test_swp_r_lock_is_carried_hash_locked_and_matches_pins`, `test_swp_r_environment_keyed_on_lock_and_child_env_cleaned`, `test_swp_r_section1_reuses_environment_and_worker_when_rerun` (executes the notebook's own kernel cell with a stand-in IPython shell; the worker runs on the test interpreter) |
| SWP-G | Fixed | Confirmed: 1 of 9 guided markers on `main`. Guided opening added (audience, Input → Model → Output, How to use this notebook with the one-pass Run all statement, roadmap with concept tags); a **Predict** prompt before Sections 4, 5, 7, 8, 9 and 11 and six collapsible **Check your reasoning** answers quoting the recorded 2026-09-15 Kaggle T4 run (baseline AP 0.000; adapted AP 0.9161 / AP50 0.9273 / AP75 0.9273; 240.6 s; pretrained IoU 0.91–0.97 with the sports ball missed). The degenerate-probe counts and per-epoch losses are not in the verification record, so those two answers are qualitative. Troubleshooting (isolated runtime, snapshot, re-running Section 8 alone, BYOD, validation errors), Change one thing, Glossary and Conclusion (your notes) sections; Sections 1–3 labelled Infrastructure and collapsed. | `tools/notebook_template.py` (`guided.opening`, prerequisites, `run_all`, Sections 4–13 markdown, closing) | `test_swp_g_guided_layer_present`, `test_swp_g_infrastructure_cells_labelled_and_collapsed`, `test_swp_g_no_leftover_placeholders`, `test_swp_g_checkpoint_answers_quote_the_recorded_run` |
| SWP-A | Not applicable | The sweep found no quality assert; Section 11's reload-parity asserts are contract checks and stay. | — | — |
| SWP-F | Not applicable | The canonical COCO pipeline `pipe` is never trained; Section 7 builds a separate re-headed `adapter` and its baseline, and Section 8 trains that. Re-running Section 8 alone continues from the adapted `adapter` (not labelled frozen); the Section 8 prose, Troubleshooting and Change one thing now say to run Section 7 first. | Section 8 markdown; closing | — |
| SWP-B | Fixed | Confirmed in the Section 13 cell: the image branch worked only through `google.colab.files.upload()` and `next(iter(uploaded.items()))` (bare `StopIteration` on a cancelled dialog). It gains a `BYOD_IMAGE_PATH` form field read by `byod_file` (Kaggle/Jupyter); the Colab upload is the guarded fallback (exactly one file; a cancelled or empty dialog, a runtime without the dialog and a missing path each get a message naming the file and the rule). The dataset branch (records supplied in code) is unchanged. | Section 13 code and markdown; `byod` header text | `test_swp_b_byod_image_path_reads_a_file_and_refusals_name_the_file`, `test_swp_b_cancelled_upload_is_named_and_one_upload_is_saved`, `test_swp_b_byod_gates_are_off_by_default_and_the_image_gate_has_a_path_field` |

## User-visible changes

- Section 1 is one collapsed infrastructure cell that builds (or reuses) `dimer_isolated_env_<lock digest>/` from the new hash-locked `tutorials/requirements-colab.lock.txt` (47 entries) and routes later cells to it; nothing is installed into the kernel and Run all needs no restart. Linux x86_64 runtimes only.
- Guided-layer markdown throughout; Sections 1–3 collapsed; Troubleshooting, Change one thing, Glossary and Conclusion sections at the end.
- Section 13 gains `BYOD_IMAGE_PATH`; BYOD image refusals are named errors instead of a bare `StopIteration`.
- The workshop notebook `DIMER_MultiModel_Closed_Set_Object_Detection_Workshop.ipynb` is untouched (`tools/build_detection_workshop.py --check` OK).
- `docs/release-verification.md`: the two lines describing the in-kernel install and restart now describe the isolated bootstrap. No status text changed.

## Verification (offline; not clean-runtime evidence)

- No model stage ran here (torch absent, Hub unreachable); no pretrained-inference evidence is claimed. Checkpoint numbers are quoted from the 2026-09-15 record in `docs/release-verification.md`.
- Stand-ins: the Section 1 kernel cell is executed with a stand-in environment and IPython shell; the Section 13 `byod_file` helper with temporary files and a fake `google.colab.files.upload`.
- `python tools/build_notebook.py --check`: OK. `python tools/build_detection_workshop.py --check`: OK. `python tools/validate_release_assets.py`: PASS. `ruff check src tests tools`: clean. `pytest` (CI deps without torch; pandas installed as CI does): 86 passed before → 96 passed after.
- `uv pip install --dry-run --require-hashes --only-binary :all: -r tutorials/requirements-colab.lock.txt` into a `uv venv --managed-python --python 3.12.12`: resolves, would install 47 packages.

## Remaining gates

- A hosted **Run all in one pass** on a fresh runtime (expected: no restart prompt; Section 1 builds the environment; a second Run all reports `'reused': True`).
- The REL12 BYOD run with the BYOD gates and path fields set.
- A full Notebook Review Framework v1 review has not been done.
