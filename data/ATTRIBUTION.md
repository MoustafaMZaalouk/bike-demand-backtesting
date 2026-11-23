# Dataset attribution and verification

- Dataset: Bike Sharing, Hadi Fanaee-T (2013).
- Repository: UCI Machine Learning Repository.
- DOI: https://doi.org/10.24432/C5W894
- Current source page: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
- Download observed on that source:
  https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip
- License: CC BY 4.0, explicitly stated in the current dataset page's
  License section, checked November 24, 2026. The HTML links to
  https://creativecommons.org/licenses/by/4.0/legalcode .
- Only `day.csv` is redistributed here, with its original bytes unchanged.
- SHA-256: `a6bcf826782d3c0fbfdcbeead17cd0884185a0dafe8ff10cd48a874ee7ba18be`.
- Observed daily records: 731, January 1, 2011 to December 31, 2012.
- The original archive README asks users to cite Fanaee-T and Gama (2013),
  Event labeling combining ensemble detectors and background knowledge,
  https://doi.org/10.1007/s13748-013-0040-3 .
- The archive README credits Capital Bikeshare rental logs and Freemeteo
  weather information. Weather is not used in this project's predictors.
- Changes: CSV unmodified. Lag/calendar features, predictions, evaluation
  tables and figures are derived outputs created for this project.

The MIT license at the repository root covers project code only, not this
CC BY 4.0 dataset. The archive README and current page use inconsistent
season-name descriptions; this project does not use that season field.
