//! Callback manager for handling multiple callbacks efficiently.

use pyo3::prelude::*;

/// Manager for multiple Python callbacks.
///
/// This allows users to register multiple callbacks that will be called
/// sequentially during optimization. Each callback can independently
/// decide whether to continue or stop the optimization.
#[pyclass(name = "CallbackManager")]
pub struct PyCallbackManager {
	callbacks: Vec<PyObject>,
	#[pyo3(get)]
	pub enabled: bool,
}

#[pymethods]
impl PyCallbackManager {
	/// Create a new callback manager.
	///
	/// Args:
	///     callbacks: Optional list of callbacks to register initially
	#[new]
	#[pyo3(signature = (callbacks=None))]
	pub fn new(callbacks: Option<Vec<PyObject>>) -> Self {
		Self {
			callbacks: callbacks.unwrap_or_default(),
			enabled: true,
		}
	}

	/// Add a callback to the manager.
	///
	/// Args:
	///     callback: The callback to add
	pub fn add_callback(&mut self, callback: PyObject) {
		self.callbacks.push(callback);
	}

	/// Remove a callback from the manager.
	///
	/// Args:
	///     callback: The callback to remove
	///
	/// Returns:
	///     bool: True if the callback was found and removed
	pub fn remove_callback(&mut self, _py: Python<'_>, callback: PyObject) -> bool {
		if let Some(pos) = self.callbacks.iter().position(|cb| cb.is(&callback)) {
			self.callbacks.remove(pos);
			true
		} else {
			false
		}
	}

	/// Clear all callbacks.
	pub fn clear(&mut self) {
		self.callbacks.clear();
	}

	/// Get the number of registered callbacks.
	#[getter]
	pub fn num_callbacks(&self) -> usize {
		self.callbacks.len()
	}

	/// Enable all callbacks.
	pub fn enable(&mut self) {
		self.enabled = true;
	}

	/// Disable all callbacks temporarily.
	pub fn disable(&mut self) {
		self.enabled = false;
	}

	/// Get the number of callbacks (for Python access).
	/// Note: Direct access to callbacks list removed due to PyObject clone limitations

	fn __repr__(&self) -> String {
		format!(
			"CallbackManager(num_callbacks={}, enabled={})",
			self.callbacks.len(),
			self.enabled
		)
	}

	fn __len__(&self) -> usize {
		self.callbacks.len()
	}
}

/// Helper to create a callback manager from Python callbacks.
#[pyfunction]
#[pyo3(signature = (*callbacks))]
pub fn create_callback_manager(callbacks: Vec<PyObject>) -> PyCallbackManager {
	PyCallbackManager::new(Some(callbacks))
}

/// Register the callback manager module.
pub fn register_callback_manager(m: &Bound<'_, PyModule>) -> PyResult<()> {
	m.add_class::<PyCallbackManager>()?;
	m.add_function(wrap_pyfunction!(create_callback_manager, m)?)?;
	Ok(())
}
