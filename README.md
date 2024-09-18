dGPMP2-ND
=====

An implementation of the DGPMP2-ND differentiable motion planner.

DGPMP2-ND is a modified and extended variant of Differentiable Gaussian Process Motion Planning DGPMP2 [1]. The code in this repo is based on [this implementation](https://github.com/mhmukadam/dgpmp2) by Mustafa Mukadam.


Installation
-----

1. Install a recent version of Python. This code was tested with Python 3.11.9 under Windows. Create a virtual environment if desired.
2. Clone this repository: `git clone git@github.com:benjaminalt/dgpmp2-nd.git`
3. Install this repository into the currently active Python environment:
```bash
cd dgpmp2-nd
pip install .
```
This will automatically install all binary dependencies, as well as [differentiable-robot-model](https://github.com/benjaminalt/differentiable-robot-model) as an additional dependency.

Example
----
A usage example and demo can be found under [demos/planning_demo.ipynb](demos/planning_demo.ipynb).

