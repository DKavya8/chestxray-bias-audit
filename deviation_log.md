# Deviation log

## 2026-08-25 — Week 2 Group A first-scan rerun

The Group A first-index-scan-per-patient rerun completed locally and its focused validation passed. The repository-wide suite still reports four pre-existing failures in `secondary_metrics.py` and `week2_task2.py`: a synthetic calibration fixture references a missing patient, two expected Group B helper functions are absent, and one synthetic Group B label fixture is non-binary. These failures were outside the Group A first-scan adapter and did not affect its generated results; they remain to be triaged before claiming a fully green repository-wide suite.
