# Aperture Optimizer

Aperture optimizer helps you find the optimal aperture from a Kepler or TESS Target Pixel File, using `lightkurve`.

**Warning: Currently not working properly...**

### TODO:

Make sure that the penalty is correct to find the best aperture.

### Installation

```
pip install apertureoptimizer
```

### Usage

Here is a basic tutorial for how to get started with `ApertureOptimizer`. You can see more examples in the demo folder.


```Python
import lightkurve as lk
from apertureoptimizer import ApertureOptimizer

# Download KOI 6.01, an exoplanet False Positive
tpf = lk.search_targetpixelfile(3248033, quarter=8).download()

# Define a corrector function
def corrector(lc):
    clc = lc.copy().flatten().remove_outliers(sigma_upper=2, sigma_lower=10)
    return clc

# Make an ApertureOptimizer Class, pass in the False Positive parameters
a = ApertureOptimizer(tpf, period=1.334104268, t0=133.701635, duration=3.0142, corrector=corrector)

# Optimize the aperture
a.optimize()

# Plot the results
a.plot_results()

# Use the best light curve as you normally would.
lc = a.best_lc
```
