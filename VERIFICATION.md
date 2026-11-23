# Local verification record

Checked November 24, 2026 on Linux / Python 3.10.12.

- Installed the pinned lock file into a second, empty virtual environment.
- Installed this project using the README command.
- Ran `make reproduce`: full training selection, full evaluation, report,
  plot, correctness tests and Ruff checks passed.
- Repeated full reproduction produced a byte-identical forecast CSV.
- Ran `make smoke` in the clean environment successfully. It uses a shorter
  date-subsampled dataset and does not replace the committed full results.
- Final test suite: 14 passed. Ruff lint and four-space format checks pass.
- Matplotlib emits upstream Pyparsing deprecation warnings with this pinned
  environment. They do not change the measured forecast results.
- Opened and visually inspected the actual generated backtest PNG: title,
  axes, legend, interval shading, monthly bars and source footer are readable.
- GitHub Actions configuration is included but was not run remotely.
- No GitHub repository, account, publication or paid service was touched.

This is a local verification record, not a claim about production reliability.
