# BRUTELIB

Brute force simulation of a special stochastic hybrid system.
 
The discrete part of the system $b \in {0, 1}^{N\times M}$ consists of $N$ two state elements organised in subgroups of size $M$. The transition rates between the two states depend on the state of the other elements in the subgroup and the external continuous variables y.
    
## INSTALL

Just install via pip
```bash
pip install .
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
