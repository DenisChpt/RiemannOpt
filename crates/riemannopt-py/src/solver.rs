use pyo3::prelude::*;

use riemannopt_core::solver::{
	Adam, AdamConfig, CGConfig, ConjugateGradient, LBFGSConfig, SGDConfig, TrustRegion,
	TrustRegionConfig, LBFGS, SGD,
};
use riemannopt_core::solver::sgd::MomentumMethod;

// ════════════════════════════════════════════════════════════════════════
//  Dynamic solver enum
// ════════════════════════════════════════════════════════════════════════

pub(crate) enum DynSolver {
	SGD(SGD<f64>),
	Adam(Adam<f64>),
	LBFGS(LBFGS<f64>),
	CG(ConjugateGradient<f64>),
	TrustRegion(TrustRegion<f64>),
}

/// Dispatch on `DynSolver`. Calls $body with $s bound to the concrete solver.
macro_rules! with_solver {
	($dyn:expr, |$s:ident| $body:expr) => {
		match $dyn {
			DynSolver::SGD(ref mut $s) => $body,
			DynSolver::Adam(ref mut $s) => $body,
			DynSolver::LBFGS(ref mut $s) => $body,
			DynSolver::CG(ref mut $s) => $body,
			DynSolver::TrustRegion(ref mut $s) => $body,
		}
	};
}

pub(crate) use with_solver;

// ════════════════════════════════════════════════════════════════════════
//  Python class
// ════════════════════════════════════════════════════════════════════════

/// Wrapper around `DynSolver`, used as parameter to `solve()`.
#[pyclass(name = "Solver")]
pub struct PySolver {
	pub(crate) inner: DynSolver,
}

// ════════════════════════════════════════════════════════════════════════
//  Factory functions (Python calls SGD(...), Adam(...), etc.)
// ════════════════════════════════════════════════════════════════════════

#[pyfunction]
#[pyo3(signature = (learning_rate=0.01, momentum=0.0, nesterov=false, gradient_clip=None))]
pub fn sgd(
	learning_rate: f64,
	momentum: f64,
	nesterov: bool,
	gradient_clip: Option<f64>,
) -> PySolver {
	let mom = if momentum > 0.0 {
		if nesterov {
			MomentumMethod::Nesterov {
				coefficient: momentum,
			}
		} else {
			MomentumMethod::Classical {
				coefficient: momentum,
			}
		}
	} else {
		MomentumMethod::None
	};

	let mut config = SGDConfig::new()
		.with_constant_step_size(learning_rate)
		.with_momentum(mom);
	if let Some(clip) = gradient_clip {
		config = config.with_gradient_clip(clip);
	}

	PySolver {
		inner: DynSolver::SGD(SGD::new(config)),
	}
}

#[pyfunction]
#[pyo3(signature = (learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8))]
pub fn adam(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) -> PySolver {
	let config = AdamConfig {
		learning_rate,
		beta1,
		beta2,
		epsilon,
		use_amsgrad: false,
		gradient_clip: None,
	};

	PySolver {
		inner: DynSolver::Adam(Adam::new(config)),
	}
}

#[pyfunction]
#[pyo3(signature = (memory=10))]
pub fn lbfgs(memory: usize) -> PySolver {
	let mut config = LBFGSConfig::default();
	config.memory_size = memory;

	PySolver {
		inner: DynSolver::LBFGS(LBFGS::new(config)),
	}
}

#[pyfunction]
pub fn cg() -> PySolver {
	PySolver {
		inner: DynSolver::CG(ConjugateGradient::new(CGConfig::default())),
	}
}

#[pyfunction]
pub fn trust_region() -> PySolver {
	PySolver {
		inner: DynSolver::TrustRegion(TrustRegion::new(TrustRegionConfig::default())),
	}
}
