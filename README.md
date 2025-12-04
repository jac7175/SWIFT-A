# SWIFT-A
Accompanying code and support material for a Shallow Water Ice Fracture Tracking and Acoustics data set, recorded on Portage Lake, Michigan in 2024.

## Description
This repository supports [data uploaded to the WDC Climate repository](#data-download).

This repo contains three folders:  
[calibrationSheets](https://github.com/jac7175/SWIFT-A/tree/main/calibrationSheets): contains calibraton sheets for all acoustic sensors used during data acquisition.  
[paperFgures](https://github.com/jac7175/SWIFT-A/tree/main/paperFigures): contains the code used to create the Figures 9-14 in the accompanying [data description paper](#data-description-paper).
[productDataSheets](https://github.com/jac7175/SWIFT-A/tree/main/productDataSheets): contains the factory product data sheets for each acoustic sensor type used during data acquisition.

## Data Download
Case, John; Brown, Daniel; Barnard, Andrew (2025). SWIFT-A: Shallow Water Ice Fracturing Tracking and Acoustics. World Data Center for Climate (WDCC) at DKRZ. https://www.wdc-climate.de/ui/entry?acronym=swifta

## Data description paper
Paper citation to be included.

## Getting Started

### Reproducing paper figures

* Data may be downloaded by following [Data Download](#data-download)
* To reproduce Figure 9, modify line 10 of [Figure9.py](paperFigures/Figures9.py) to point to passive file: 030224_105346.h5
* To reproduce Figure 10, modify line 10 of [Figure10.py](paperFigures/Figures10.py) to point to passive file: 022624_191850.h5
* To reproduce Figure 11, modify line 10 of [Figure11.py](paperFigures/Figures11.py) to point to impact file: A7.h5
* To reproduce Figure 12, modify line 10 of [Figure12.py](paperFigures/Figures12.py) to point to impact file: C8.h5

## Authors

John Case, jackcase97@gmail.com
Andrew Barnard
Daniel Brown

## Version History

* 0.1
    * Initial Release

## License

This project is licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)] License - see the LICENSE.md file for details

