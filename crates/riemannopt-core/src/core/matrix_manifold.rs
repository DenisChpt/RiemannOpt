//! Matrix manifold trait for manifolds that naturally work with matrix representations.
//!
//! This module provides a trait for manifolds whose elements are naturally represented
//! as matrices (e.g., Stiefel, Grassmann, SPD manifolds). It acts as a marker trait
//! to indicate that a manifold uses DMatrix as its Point type.

use crate::{manifold::Manifold, types::Scalar};
use nalgebra::DMatrix;

/// Trait for manifolds whose elements are naturally represented as matrices.
///
/// This is a marker trait that extends the base `Manifold` trait with the constraint
/// that the Point type must be `DMatrix<T>`. It provides matrix-specific functionality
/// without duplicating the methods from the base `Manifold` trait.
pub trait MatrixManifold<T: Scalar>: Manifold<T, Point = DMatrix<T>> {
	/// Get the dimensions of the matrix representation (rows, columns).
	fn matrix_dims(&self) -> (usize, usize);
	// Add other matrix-specific methods here if they are REALLY specific
	// to matrices and not generalizable in the Manifold trait.
}
