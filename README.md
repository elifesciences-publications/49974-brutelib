# BRUTELIB

Brute force simulation of a special stochastic hybrid system.
 
The discrete part of the system $b \in {0, 1}^{N\times M}$ consists of $N$ two state elements organised in subgroups of size $M$. The transition rates between the two states depend on the state of the other elements in the subgroup and the external continuous variables y.
    
## INSTALL

Install via pip, either by first cloning the repository
```bash
pip install .
```
or if you just want the package, use
```bash
pip install git+https://itbgit.biologie.hu-berlin.de/cooperativity/brutelib.git 
```

**Note**: You can also add the package to a conda environment
```yaml
name: my_env
dependencies:
  - ...
  - pip:
    - "git+https://itbgit.biologie.hu-berlin.de/cooperativity/brutelib.git"
```
## TESTS

```bash
pytest
```
## EXAMPLES 

A simple neuron equipped with a small number of cooperative ion channels.
```bash
python examples/izhekevich_neuron_with_cooperative_calcium_channels.py 

```
