"""RiemannOpt — High-performance Riemannian optimization."""

from riemannopt._native import (
    # Manifold
    Manifold,
    sphere,
    stiefel,
    grassmann,
    euclidean,
    spd,
    hyperbolic,
    oblique,
    # Solvers (factory functions returning Solver)
    Solver,
    sgd as SGD,
    adam as Adam,
    lbfgs as LBFGS,
    cg as CG,
    trust_region as TrustRegion,
    # AutoDiff
    AdSession,
    ScalarVar,
    VectorVar,
    MatrixVar,
    # Problem
    Problem,
    rayleigh_quotient,
    quadratic_cost,
    rosenbrock,
    brockett_cost,
    procrustes,
    # Preconditioner
    Preconditioner,
    # Result + Stopping
    SolverResult,
    StoppingCriterion,
    # Top-level
    solve,
)

__all__ = [
    "Manifold", "sphere", "stiefel", "grassmann", "euclidean",
    "spd", "hyperbolic", "oblique",
    "SGD", "Adam", "LBFGS", "CG", "TrustRegion", "Solver",
    "AdSession", "ScalarVar", "VectorVar", "MatrixVar",
    "Problem", "rayleigh_quotient", "quadratic_cost", "rosenbrock",
    "brockett_cost", "procrustes", "Preconditioner",
    "SolverResult", "StoppingCriterion",
    "solve",
]
