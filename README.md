# GEOMETRIC ASPECTS OF DETECTING GRASSLAND MOWING IN KRKONOŠE MOUNTAINS BASED ON SENTINEL-1 COHERENCE

Code developed during my Master's thesis which monitors mowing from time series of C-band SAR (Sentinel-1) data in Krkonose National Park, Czechia.

## Thesis abstract

Grassland mowing is a common management practice used in European grasslands for livestock fodder production and to enhance biodiversity. To support a less intensive use of grasslands, public agencies look for a reliable way to monitor the management performed on the grasslands. Satellite remote sensing is a key tool for monitoring over large areas, with SAR remote sensing being especially useful in areas with high cloud cover.

However, grassland monitoring using SAR in complex terrain is not fully understood and may come with challenges related to topography and sensor geometry. To explore these potential challenges, this thesis detected mowing events using a high-resolution DEM for precise coregistration and terrain correction of Sentinel-1 SAR imagery. Effect of local incidence angle on detection accuracy from interferometric coherence was also explored. The hypotheses were tested on 61 grassland plots in Krkonoše mountains, Czechia.

Detection accuracies in this thesis were higher than in previous studies when only considering SAR detections. The improvement was most likely caused by counting detections from individual orbits to assess the certainty of each detection. A deeper analysis showed that using a high-resolution DEM led to a horizontal shift in computed coherence, but the shift had no notable effect on detection accuracy. Topography and local incidence angle have also not shown any clear relationship with detection accuracy.

The results suggest that the current Sentinel-1 coherence processing techniques are suitable for mowing detection even in mountainous terrain, and further developments should focus on other aspects of the system, such as SAR-optical fusion and the detection algorithm.

## Usage

**01_data_download** is used for downloading

**figure_creation/** contains scripts used for various visualisations used in the study.

## License

This code is released under the MIT license.