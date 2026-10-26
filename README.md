# PhaseField-molvol
Uses FEniCS version 1.9 to model phase separation in the LiFePO4 battery electrode material using either a fixed reference frame or one where variations in molar volume leads to local dilation/contraction.

Models A-C assume a uniform, non-dilating lattice-constrained reference frame.

Model A: considers only chemical free energy

Model B: A + elastic energy

Model C: A + displacement field

Model D: chemical and elastic effects within a dilating reference frame

Install FenICS version 1.9

The input parameters are set in config.py

To run model X type the following in the command line:
python modelX.py solution_files
