import numpy as np
from ase.build import molecule
from ase.calculators.lj import LennardJones

from sella.optimize.irc import IRC


def test_irc_peskwargs_preserved():
    """A non-None peskwargs must be stored.

    Regression: self.peskwargs used to be assigned only in the
    ``if peskwargs is None`` branch, so passing an explicit peskwargs left
    the attribute unset and irun() crashed with AttributeError.
    """
    atoms = molecule('H2O')
    atoms.calc = LennardJones()
    opt = IRC(atoms, peskwargs={'gamma': 0.2}, logfile=None)
    assert opt.peskwargs == {'gamma': 0.2}


def test_irc_default_peskwargs():
    """The default path still derives peskwargs from gamma."""
    atoms = molecule('H2O')
    atoms.calc = LennardJones()
    opt = IRC(atoms, gamma=0.3, logfile=None)
    assert opt.peskwargs == {'gamma': 0.3}
