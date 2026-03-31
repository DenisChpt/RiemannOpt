//! Dual numbers for forward-mode automatic differentiation.
//!
//! A dual number `x + ε·ẋ` carries both a primal value and a tangent
//! (directional derivative).  Arithmetic on duals propagates derivatives
//! via the chain rule automatically.
//!
//! Used by the forward-over-reverse HVP computation:
//! running a forward pass with dual scalars and then a backward pass
//! on the tangent component yields the Hessian-vector product in a
//! single evaluation.

use std::ops;

use riemannopt_core::linalg::RealScalar;

/// A dual number `val + dot·ε` where `ε² = 0`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Dual<T: RealScalar> {
	/// Primal (function value).
	pub val: T,
	/// Tangent (directional derivative).
	pub dot: T,
}

impl<T: RealScalar> Dual<T> {
	/// Creates a dual number with explicit primal and tangent.
	#[inline]
	pub fn new(val: T, dot: T) -> Self {
		Self { val, dot }
	}

	/// Creates a constant (tangent = 0).
	#[inline]
	pub fn constant(val: T) -> Self {
		Self {
			val,
			dot: T::zero(),
		}
	}

	/// Creates a variable with tangent = 1 (for differentiation).
	#[inline]
	pub fn variable(val: T) -> Self {
		Self { val, dot: T::one() }
	}

	// ── Transcendental functions ─────────────────────────────────────

	#[inline]
	pub fn exp(self) -> Self {
		let e = self.val.exp();
		Self {
			val: e,
			dot: self.dot * e,
		}
	}

	#[inline]
	pub fn ln(self) -> Self {
		Self {
			val: self.val.ln(),
			dot: self.dot / self.val,
		}
	}

	#[inline]
	pub fn sqrt(self) -> Self {
		let s = self.val.sqrt();
		let two = T::one() + T::one();
		Self {
			val: s,
			dot: self.dot / (two * s),
		}
	}

	#[inline]
	pub fn sin(self) -> Self {
		Self {
			val: self.val.sin(),
			dot: self.dot * self.val.cos(),
		}
	}

	#[inline]
	pub fn cos(self) -> Self {
		Self {
			val: self.val.cos(),
			dot: T::zero() - self.dot * self.val.sin(),
		}
	}

	#[inline]
	pub fn abs(self) -> Self {
		Self {
			val: self.val.abs(),
			dot: self.dot * self.val.signum(),
		}
	}

	#[inline]
	pub fn powf(self, exp: Self) -> Self {
		let val = self.val.powf(exp.val);
		// d/dt f^g = f^g * (g' * ln(f) + g * f'/f)
		let dot = val * (exp.dot * self.val.ln() + exp.val * self.dot / self.val);
		Self { val, dot }
	}

	#[inline]
	pub fn signum(self) -> Self {
		Self {
			val: self.val.signum(),
			dot: T::zero(),
		}
	}
}

// ── Arithmetic operator implementations ─────────────────────────────

impl<T: RealScalar> ops::Add for Dual<T> {
	type Output = Self;
	#[inline]
	fn add(self, rhs: Self) -> Self {
		Self {
			val: self.val + rhs.val,
			dot: self.dot + rhs.dot,
		}
	}
}

impl<T: RealScalar> ops::Sub for Dual<T> {
	type Output = Self;
	#[inline]
	fn sub(self, rhs: Self) -> Self {
		Self {
			val: self.val - rhs.val,
			dot: self.dot - rhs.dot,
		}
	}
}

impl<T: RealScalar> ops::Mul for Dual<T> {
	type Output = Self;
	#[inline]
	fn mul(self, rhs: Self) -> Self {
		Self {
			val: self.val * rhs.val,
			dot: self.dot * rhs.val + self.val * rhs.dot,
		}
	}
}

impl<T: RealScalar> ops::Div for Dual<T> {
	type Output = Self;
	#[inline]
	fn div(self, rhs: Self) -> Self {
		let inv = T::one() / rhs.val;
		Self {
			val: self.val * inv,
			dot: (self.dot * rhs.val - self.val * rhs.dot) * inv * inv,
		}
	}
}

impl<T: RealScalar> ops::Neg for Dual<T> {
	type Output = Self;
	#[inline]
	fn neg(self) -> Self {
		Self {
			val: T::zero() - self.val,
			dot: T::zero() - self.dot,
		}
	}
}

impl<T: RealScalar> ops::AddAssign for Dual<T> {
	#[inline]
	fn add_assign(&mut self, rhs: Self) {
		self.val = self.val + rhs.val;
		self.dot = self.dot + rhs.dot;
	}
}

impl<T: RealScalar> ops::SubAssign for Dual<T> {
	#[inline]
	fn sub_assign(&mut self, rhs: Self) {
		self.val = self.val - rhs.val;
		self.dot = self.dot - rhs.dot;
	}
}

impl<T: RealScalar> ops::MulAssign for Dual<T> {
	#[inline]
	fn mul_assign(&mut self, rhs: Self) {
		let new_dot = self.dot * rhs.val + self.val * rhs.dot;
		self.val = self.val * rhs.val;
		self.dot = new_dot;
	}
}
