# Hyperspectral visualization

This code is used to visualize data where a 1D signal is acquired for a range of 2D points (thus 3D data). In the present case it is implemented for the reflection spectra of a dual-gated field-effect device 
where the 2D points stand for different applied bottom and top gate voltages, see the figure below:

![device](https://github.com/borya-polovnikov/hyperspectral_visualization/assets/147932035/0a9e3d9e-eedc-4ee6-8f26-8d1b56e13978)

In the present script, the third dimension (i.e. the spectral signal) is projected to the maximum along its axes, which is then plotted in 2D map on the left. 
Individual pixels within this map can be addressed by the mouse to display the whole spectral signal in a separate axis.

The script includes three classes:

  1. An abstract class creating interactive cursors for matplotlib figures, used to make the 2D map interactive.
  2. A visualizing class for 2D heatmaps that are used to plot linecuts through the 2D map of the 3D data.
  3. The main GateMap class that bundles all the data and methods needed to visualize the 3D dataset.

Note that matpotlib version 3.5 or higher is needed to allow for the RangeSlider widget that is used to select the spectral range to be considered.


# Interface


https://github.com/borya-polovnikov/hyperspectral_visualization_tool/assets/147932035/d5130cc3-825e-4395-aea3-356b79121765

