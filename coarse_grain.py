import biotite.structure as btstr
from biotite.structure.io.pdb import PDBFile
from collections import namedtuple
import numpy as np
from typing import List
import r3

CG1 = ('C', 'CA', 'CB', 'N')
CG2 = ('C', 'CA', 'O')

"""Lee et al. (2022) Table 3 Amino Acid CG scheme
Atom and Residue naming conventions from PDB 
"""
scheme = {
    'ALA': [CG1, CG2],
    'ARG': [CG1, CG2, ('CB', 'CG', 'CD'), ('NE', 'NH1', 'NH2', 'CZ')],
    'ASN': [CG1, CG2, ('CG', 'ND2', 'OD1')],
    'ASP': [CG1, CG2, ('CG', 'OD1', 'OD2')], 
    'CYS': [CG1, CG2, ('CA', 'CB', 'SG')], 
    'GLN': [CG1, CG2, ('CG', 'CD', 'OE1', 'NE2')], 
    'GLU': [CG1, CG2, ('CG', 'CD', 'OE1', 'OE2')], 
    'GLY': [('C', 'CA', 'N'), CG2], 
    'HIS': [CG1, CG2, ('CG', 'CD2', 'CE1', 'ND1', 'NE2')], 
    'ILE': [CG1, CG2, ('CB', 'CG1', 'CG2'), ('CB', 'CG1', 'CD1')], 
    'LEU': [CG1, CG2, ('CG', 'CD1', 'CD2')], 
    'LYS': [CG1, CG2, ('CB', 'CG', 'CD'), ('CD', 'CE', 'NZ')], 
    'MET': [CG1, CG2, ('CG', 'CE', 'SD')], 
    'PHE': [CG1, CG2, ('CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ')], 
    'PRO': [CG1, CG2, ('CB', 'CG', 'CD')], 
    'SER': [CG1, CG2, ('CA', 'CB', 'OG')], 
    'THR': [CG1, CG2, ('CB', 'CG2', 'OG1')],
    'TRP': [CG1, CG2, ('CG', 'CD1', 'CD2', 'CE3', 'CZ2', 'CZ3', 'CH2', 'NE1')], 
    'TYR': [CG1, CG2, ('CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH')], 
    'VAL': [CG1, CG2, ('CB', 'CG1', 'CG2')] 
}

def read_pdb(pdb: str) -> btstr.AtomArray:
    structure = PDBFile.read(pdb).get_structure(1)
    return structure

def coarse_grain(atoms: btstr.AtomArray) -> List[namedtuple]:
    CG = []
    residues = np.unique(atoms[~atoms.hetero].res_id)
    for res in residues:
        res_atoms = atoms[atoms.res_id == res]
        nodes = scheme[res_atoms.res_name[0]]
        cg_nodes = []
        for node in nodes:
            cg_nodes.append(tuple(map(lambda x: res_atoms[res_atoms.atom_name == x][0], node)))
        CG.append(cg_nodes)
    return CG
