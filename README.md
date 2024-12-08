# O3O10
Computer-assisted proof of the existence of the O(3) x O(10) non-round Einstein metric, accompanying the paper found [here](https://arxiv.org/abs/2412.01184). Arbitrary precision operations are conducted using the [FLINT](https://python-flint.readthedocs.io/en/latest/) package. To run this code, ensure you are have the latest version of `Python` installed. You can install required packages using the command

```
pip install -r requirements.txt
```

Jupyter notebooks are used to interface with the implementation. These are as follows:

- `results.ipynb`: Primary Jupyter notebook to generate high-precision solutions and compute all relevant quantities in the associated manuscript.
- `proof.ipynb`: A Jupyter notebook to verify computed quantities in the proof.
- `plots.ipynb`: A Jupyter notebook to plot $(A(\alpha),\Omega(\omega))$ for a variety of different $d_1$ and $d_2$.

The implementation itself is contained in four Python scripts:

- `cohom1.py`: The implementation of the high-precision ODE solver for $\eta$ and $\zeta$ (contained in the class `Eta`) and $A$ and $\Omega$ (contained in the class `AlphaOmega`). This file also contains a number of helper functions to generate the required solutions in the manuscript.
- `linearized.py`: Implementation of the differential of $\eta$ and $\zeta$ with respect to $\alpha$ using finite differences (contained in the class `DiffEta`). Contains a few helper functions as well.
- `arb_cheby.py`: Arbitrary-precision implementations of the Chebyshev polynomials with appropriate in-basis manipulations.
- `arb_roots.py`: Root-finding procedures in arbitrary precision. 

Figures are contained in the `figs` folder. MATLAB scripts to plot $\eta$ and $\zeta$ are contained in the `matlab` folder. 

All precomputed quantities are to be stored in the `data` folder. **You do not need to compute these quantities if you download the precomputed data files in the Releases tab.**