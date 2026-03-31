//! `AutoDiff` graph recording and replay for Python.
//!
//! Python users build a computation graph via `PyAdSession` method calls.
//! Each call appends a `GraphOp` to a `RecordedGraph`. At `build_problem()`,
//! the graph is frozen into a Rust closure that replays the ops on a fresh
//! `AdSession` — no Python objects captured, no GIL needed.

use numpy::PyReadonlyArray1;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;

use riemannopt_autodiff::{AdSession, SVar, VVar, MVar};
use riemannopt_core::linalg::DefaultBackend;

use crate::convert::{numpy_1d_to_col, numpy_2d_to_mat, Mat64, Vec64};
use crate::problem::PyProblem;

type B = DefaultBackend;

// ════════════════════════════════════════════════════════════════════════
//  Variable handles (opaque to Python)
// ════════════════════════════════════════════════════════════════════════

#[pyclass(name = "ScalarVar", frozen, from_py_object)]
#[derive(Clone, Copy)]
pub struct ScalarVar(pub(crate) u32);

#[pyclass(name = "VectorVar", frozen, from_py_object)]
#[derive(Clone, Copy)]
pub struct VectorVar(pub(crate) u32);

#[pyclass(name = "MatrixVar", frozen, from_py_object)]
#[derive(Clone, Copy)]
pub struct MatrixVar(pub(crate) u32);

#[pymethods]
impl ScalarVar {
	fn __repr__(&self) -> String {
		format!("ScalarVar({})", self.0)
	}
}
#[pymethods]
impl VectorVar {
	fn __repr__(&self) -> String {
		format!("VectorVar({})", self.0)
	}
}
#[pymethods]
impl MatrixVar {
	fn __repr__(&self) -> String {
		format!("MatrixVar({})", self.0)
	}
}

// ════════════════════════════════════════════════════════════════════════
//  Recorded computation graph
// ════════════════════════════════════════════════════════════════════════

/// A single operation in the recorded graph.
#[derive(Clone)]
pub(crate) enum GraphOp {
	InputVector { len: usize },
	InputMatrix { rows: usize, cols: usize },
	ConstScalar(usize),
	ConstVector(usize),
	ConstMatrix(usize),
	// Scalar ops
	AddS(u32, u32), SubS(u32, u32), MulS(u32, u32), DivS(u32, u32),
	NegS(u32), ExpS(u32), LogS(u32), SqrtS(u32),
	SinS(u32), CosS(u32), AbsS(u32), PowS(u32, u32),
	// Vector ops
	AddV(u32, u32), SubV(u32, u32), NegV(u32), ComponentMulV(u32, u32),
	ScaleV(u32, u32), DotV(u32, u32), NormV(u32), NormSqV(u32), SumV(u32),
	// Matrix ops
	AddM(u32, u32), SubM(u32, u32), NegM(u32), ScaleM(u32, u32),
	MatMul(u32, u32), MatVec(u32, u32), TransposeM(u32),
	TraceM(u32), FrobDotM(u32, u32),
	// Fused
	LinearLayer(u32, u32, u32), QuadForm(u32, u32),
}

/// Handle type tag for tracking which arena a handle refers to.
#[derive(Clone, Copy)]
enum HandleKind {
	Scalar,
	Vector,
	Matrix,
}

/// A recorded graph that can be replayed on any `AdSession`.
#[derive(Clone)]
pub(crate) struct RecordedGraph {
	pub(crate) ops: Vec<GraphOp>,
	handle_kinds: Vec<HandleKind>,
	pub(crate) scalars: Vec<f64>,
	pub(crate) vectors: Vec<Vec64>,
	pub(crate) matrices: Vec<Mat64>,
}

impl RecordedGraph {
	fn new() -> Self {
		Self {
			ops: Vec::new(),
			handle_kinds: Vec::new(),
			scalars: Vec::new(),
			vectors: Vec::new(),
			matrices: Vec::new(),
		}
	}

	fn push_scalar_handle(&mut self) -> u32 {
		let h = u32::try_from(self.handle_kinds.len()).expect("too many elements");
		self.handle_kinds.push(HandleKind::Scalar);
		h
	}

	fn push_vector_handle(&mut self) -> u32 {
		let h = u32::try_from(self.handle_kinds.len()).expect("too many elements");
		self.handle_kinds.push(HandleKind::Vector);
		h
	}

	fn push_matrix_handle(&mut self) -> u32 {
		let h = u32::try_from(self.handle_kinds.len()).expect("too many elements");
		self.handle_kinds.push(HandleKind::Matrix);
		h
	}
}

// ════════════════════════════════════════════════════════════════════════
//  Replay logic
// ════════════════════════════════════════════════════════════════════════

/// Internal var storage for replay.
enum RVar {
	S(SVar),
	V(VVar),
	M(MVar),
}

macro_rules! rs {
	($vars:expr, $h:expr) => {
		match &$vars[$h as usize] { RVar::S(v) => *v, _ => panic!("expected scalar") }
	};
}
macro_rules! rv {
	($vars:expr, $h:expr) => {
		match &$vars[$h as usize] { RVar::V(v) => *v, _ => panic!("expected vector") }
	};
}
macro_rules! rm {
	($vars:expr, $h:expr) => {
		match &$vars[$h as usize] { RVar::M(v) => *v, _ => panic!("expected matrix") }
	};
}

/// Common replay logic for all ops except inputs.
fn replay_op(
	op: &GraphOp,
	graph: &RecordedGraph,
	session: &mut AdSession<f64, B>,
	vars: &mut Vec<RVar>,
	last_scalar: &mut Option<SVar>,
) {
	match op {
		GraphOp::InputVector { .. } | GraphOp::InputMatrix { .. } => {
			// Handled by caller
		}
		GraphOp::ConstScalar(idx) => { let r = session.constant_scalar(graph.scalars[*idx]); vars.push(RVar::S(r)); }
		GraphOp::ConstVector(idx) => { let r = session.constant_vector(&graph.vectors[*idx]); vars.push(RVar::V(r)); }
		GraphOp::ConstMatrix(idx) => { let r = session.constant_matrix(&graph.matrices[*idx]); vars.push(RVar::M(r)); }
		// Scalar ops
		GraphOp::AddS(a, b) => { let r = session.add_s(rs!(vars, *a), rs!(vars, *b)); *last_scalar = Some(r); vars.push(RVar::S(r)); }
		GraphOp::SubS(a, b) => { let r = session.sub_s(rs!(vars, *a), rs!(vars, *b)); *last_scalar = Some(r); vars.push(RVar::S(r)); }
		GraphOp::MulS(a, b) => { let r = session.mul_s(rs!(vars, *a), rs!(vars, *b)); *last_scalar = Some(r); vars.push(RVar::S(r)); }
		GraphOp::DivS(a, b) => { let r = session.div_s(rs!(vars, *a), rs!(vars, *b)); *last_scalar = Some(r); vars.push(RVar::S(r)); }
		GraphOp::NegS(a) => { let r = session.neg_s(rs!(vars, *a)); *last_scalar = Some(r); vars.push(RVar::S(r)); }
		GraphOp::ExpS(a) => { let r = session.exp_s(rs!(vars, *a)); *last_scalar = Some(r); vars.push(RVar::S(r)); }
		GraphOp::LogS(a) => { let r = session.log_s(rs!(vars, *a)); *last_scalar = Some(r); vars.push(RVar::S(r)); }
		GraphOp::SqrtS(a) => { let r = session.sqrt_s(rs!(vars, *a)); *last_scalar = Some(r); vars.push(RVar::S(r)); }
		GraphOp::SinS(a) => { let r = session.sin_s(rs!(vars, *a)); *last_scalar = Some(r); vars.push(RVar::S(r)); }
		GraphOp::CosS(a) => { let r = session.cos_s(rs!(vars, *a)); *last_scalar = Some(r); vars.push(RVar::S(r)); }
		GraphOp::AbsS(a) => { let r = session.abs_s(rs!(vars, *a)); *last_scalar = Some(r); vars.push(RVar::S(r)); }
		GraphOp::PowS(a, b) => { let r = session.pow_s(rs!(vars, *a), rs!(vars, *b)); *last_scalar = Some(r); vars.push(RVar::S(r)); }
		// Vector ops
		GraphOp::AddV(a, b) => { vars.push(RVar::V(session.add_v(rv!(vars, *a), rv!(vars, *b)))); }
		GraphOp::SubV(a, b) => { vars.push(RVar::V(session.sub_v(rv!(vars, *a), rv!(vars, *b)))); }
		GraphOp::NegV(a) => { vars.push(RVar::V(session.neg_v(rv!(vars, *a)))); }
		GraphOp::ComponentMulV(a, b) => { vars.push(RVar::V(session.component_mul_v(rv!(vars, *a), rv!(vars, *b)))); }
		GraphOp::ScaleV(s, v) => { vars.push(RVar::V(session.scale_v(rs!(vars, *s), rv!(vars, *v)))); }
		GraphOp::DotV(a, b) => { let r = session.dot(rv!(vars, *a), rv!(vars, *b)); *last_scalar = Some(r); vars.push(RVar::S(r)); }
		GraphOp::NormV(a) => { let r = session.norm_v(rv!(vars, *a)); *last_scalar = Some(r); vars.push(RVar::S(r)); }
		GraphOp::NormSqV(a) => { let r = session.norm_sq_v(rv!(vars, *a)); *last_scalar = Some(r); vars.push(RVar::S(r)); }
		GraphOp::SumV(a) => { let r = session.sum_v(rv!(vars, *a)); *last_scalar = Some(r); vars.push(RVar::S(r)); }
		// Matrix ops
		GraphOp::AddM(a, b) => { vars.push(RVar::M(session.add_m(rm!(vars, *a), rm!(vars, *b)))); }
		GraphOp::SubM(a, b) => { vars.push(RVar::M(session.sub_m(rm!(vars, *a), rm!(vars, *b)))); }
		GraphOp::NegM(a) => { vars.push(RVar::M(session.neg_m(rm!(vars, *a)))); }
		GraphOp::ScaleM(s, m) => { vars.push(RVar::M(session.scale_m(rs!(vars, *s), rm!(vars, *m)))); }
		GraphOp::MatMul(a, b) => { vars.push(RVar::M(session.mat_mul(rm!(vars, *a), rm!(vars, *b)))); }
		GraphOp::MatVec(m, v) => { vars.push(RVar::V(session.mat_vec(rm!(vars, *m), rv!(vars, *v)))); }
		GraphOp::TransposeM(a) => { vars.push(RVar::M(session.transpose_m(rm!(vars, *a)))); }
		GraphOp::TraceM(m) => { let r = session.trace_m(rm!(vars, *m)); *last_scalar = Some(r); vars.push(RVar::S(r)); }
		GraphOp::FrobDotM(a, b) => { let r = session.frob_dot(rm!(vars, *a), rm!(vars, *b)); *last_scalar = Some(r); vars.push(RVar::S(r)); }
		// Fused
		GraphOp::LinearLayer(a, x, b) => { vars.push(RVar::V(session.linear_layer(rm!(vars, *a), rv!(vars, *x), rv!(vars, *b)))); }
		GraphOp::QuadForm(x, a) => { let r = session.quad_form(rv!(vars, *x), rm!(vars, *a)); *last_scalar = Some(r); vars.push(RVar::S(r)); }
	}
}

/// Replay for vector-point problems. The first `InputVector` op uses the
/// pre-registered `input_vvar` (created by `AutoDiffProblem`).
pub(crate) fn replay_vec(
	graph: &RecordedGraph,
	session: &mut AdSession<f64, B>,
	input_vvar: VVar,
) -> SVar {
	let mut vars = Vec::with_capacity(graph.handle_kinds.len());
	let mut last_scalar: Option<SVar> = None;
	let mut input_consumed = false;

	for op in &graph.ops {
		match op {
			GraphOp::InputVector { .. } if !input_consumed => {
				vars.push(RVar::V(input_vvar));
				input_consumed = true;
			}
			GraphOp::InputVector { len } => {
				let v = Vec64::zeros(*len);
				let var = session.input_vector(&v);
				vars.push(RVar::V(var));
			}
			GraphOp::InputMatrix { rows, cols } => {
				let m = Mat64::zeros(*rows, *cols);
				let var = session.input_matrix(&m);
				vars.push(RVar::M(var));
			}
			_ => replay_op(op, graph, session, &mut vars, &mut last_scalar),
		}
	}

	last_scalar.expect("graph must produce at least one scalar (the loss)")
}

/// Replay for matrix-point problems. The first `InputMatrix` op uses the
/// pre-registered `input_mvar`.
pub(crate) fn replay_mat(
	graph: &RecordedGraph,
	session: &mut AdSession<f64, B>,
	input_mvar: MVar,
) -> SVar {
	let mut vars = Vec::with_capacity(graph.handle_kinds.len());
	let mut last_scalar: Option<SVar> = None;
	let mut input_consumed = false;

	for op in &graph.ops {
		match op {
			GraphOp::InputMatrix { .. } if !input_consumed => {
				vars.push(RVar::M(input_mvar));
				input_consumed = true;
			}
			GraphOp::InputVector { len } => {
				let v = Vec64::zeros(*len);
				let var = session.input_vector(&v);
				vars.push(RVar::V(var));
			}
			GraphOp::InputMatrix { rows, cols } => {
				let m = Mat64::zeros(*rows, *cols);
				let var = session.input_matrix(&m);
				vars.push(RVar::M(var));
			}
			_ => replay_op(op, graph, session, &mut vars, &mut last_scalar),
		}
	}

	last_scalar.expect("graph must produce at least one scalar (the loss)")
}

// ════════════════════════════════════════════════════════════════════════
//  Python-facing session
// ════════════════════════════════════════════════════════════════════════

#[pyclass(name = "AdSession")]
pub struct PyAdSession {
	graph: RecordedGraph,
}

#[pymethods]
impl PyAdSession {
	#[new]
	fn new() -> Self {
		Self { graph: RecordedGraph::new() }
	}

	// ── Inputs ───────────────────────────────────────────────────────

	fn input_vector(&mut self, n: usize) -> VectorVar {
		let h = self.graph.push_vector_handle();
		self.graph.ops.push(GraphOp::InputVector { len: n });
		VectorVar(h)
	}

	fn input_matrix(&mut self, rows: usize, cols: usize) -> MatrixVar {
		let h = self.graph.push_matrix_handle();
		self.graph.ops.push(GraphOp::InputMatrix { rows, cols });
		MatrixVar(h)
	}

	// ── Constants ────────────────────────────────────────────────────

	fn constant_scalar(&mut self, val: f64) -> ScalarVar {
		let idx = self.graph.scalars.len();
		self.graph.scalars.push(val);
		let h = self.graph.push_scalar_handle();
		self.graph.ops.push(GraphOp::ConstScalar(idx));
		ScalarVar(h)
	}

	fn constant_vector(&mut self, data: PyReadonlyArray1<'_, f64>) -> VectorVar {
		let vec = numpy_1d_to_col(data);
		let idx = self.graph.vectors.len();
		self.graph.vectors.push(vec);
		let h = self.graph.push_vector_handle();
		self.graph.ops.push(GraphOp::ConstVector(idx));
		VectorVar(h)
	}

	fn constant_matrix(&mut self, data: PyReadonlyArray2<'_, f64>) -> MatrixVar {
		let mat = numpy_2d_to_mat(data);
		let idx = self.graph.matrices.len();
		self.graph.matrices.push(mat);
		let h = self.graph.push_matrix_handle();
		self.graph.ops.push(GraphOp::ConstMatrix(idx));
		MatrixVar(h)
	}

	// ── Scalar ops ───────────────────────────────────────────────────

	fn add_s(&mut self, a: ScalarVar, b: ScalarVar) -> ScalarVar { let h = self.graph.push_scalar_handle(); self.graph.ops.push(GraphOp::AddS(a.0, b.0)); ScalarVar(h) }
	fn sub_s(&mut self, a: ScalarVar, b: ScalarVar) -> ScalarVar { let h = self.graph.push_scalar_handle(); self.graph.ops.push(GraphOp::SubS(a.0, b.0)); ScalarVar(h) }
	fn mul_s(&mut self, a: ScalarVar, b: ScalarVar) -> ScalarVar { let h = self.graph.push_scalar_handle(); self.graph.ops.push(GraphOp::MulS(a.0, b.0)); ScalarVar(h) }
	fn div_s(&mut self, a: ScalarVar, b: ScalarVar) -> ScalarVar { let h = self.graph.push_scalar_handle(); self.graph.ops.push(GraphOp::DivS(a.0, b.0)); ScalarVar(h) }
	fn neg_s(&mut self, a: ScalarVar) -> ScalarVar { let h = self.graph.push_scalar_handle(); self.graph.ops.push(GraphOp::NegS(a.0)); ScalarVar(h) }
	fn exp_s(&mut self, a: ScalarVar) -> ScalarVar { let h = self.graph.push_scalar_handle(); self.graph.ops.push(GraphOp::ExpS(a.0)); ScalarVar(h) }
	fn log_s(&mut self, a: ScalarVar) -> ScalarVar { let h = self.graph.push_scalar_handle(); self.graph.ops.push(GraphOp::LogS(a.0)); ScalarVar(h) }
	fn sqrt_s(&mut self, a: ScalarVar) -> ScalarVar { let h = self.graph.push_scalar_handle(); self.graph.ops.push(GraphOp::SqrtS(a.0)); ScalarVar(h) }
	fn sin_s(&mut self, a: ScalarVar) -> ScalarVar { let h = self.graph.push_scalar_handle(); self.graph.ops.push(GraphOp::SinS(a.0)); ScalarVar(h) }
	fn cos_s(&mut self, a: ScalarVar) -> ScalarVar { let h = self.graph.push_scalar_handle(); self.graph.ops.push(GraphOp::CosS(a.0)); ScalarVar(h) }
	fn abs_s(&mut self, a: ScalarVar) -> ScalarVar { let h = self.graph.push_scalar_handle(); self.graph.ops.push(GraphOp::AbsS(a.0)); ScalarVar(h) }
	fn pow_s(&mut self, a: ScalarVar, b: ScalarVar) -> ScalarVar { let h = self.graph.push_scalar_handle(); self.graph.ops.push(GraphOp::PowS(a.0, b.0)); ScalarVar(h) }

	// ── Vector ops ───────────────────────────────────────────────────

	fn add_v(&mut self, a: VectorVar, b: VectorVar) -> VectorVar { let h = self.graph.push_vector_handle(); self.graph.ops.push(GraphOp::AddV(a.0, b.0)); VectorVar(h) }
	fn sub_v(&mut self, a: VectorVar, b: VectorVar) -> VectorVar { let h = self.graph.push_vector_handle(); self.graph.ops.push(GraphOp::SubV(a.0, b.0)); VectorVar(h) }
	fn neg_v(&mut self, a: VectorVar) -> VectorVar { let h = self.graph.push_vector_handle(); self.graph.ops.push(GraphOp::NegV(a.0)); VectorVar(h) }
	fn component_mul_v(&mut self, a: VectorVar, b: VectorVar) -> VectorVar { let h = self.graph.push_vector_handle(); self.graph.ops.push(GraphOp::ComponentMulV(a.0, b.0)); VectorVar(h) }
	fn scale_v(&mut self, s: ScalarVar, v: VectorVar) -> VectorVar { let h = self.graph.push_vector_handle(); self.graph.ops.push(GraphOp::ScaleV(s.0, v.0)); VectorVar(h) }
	fn dot(&mut self, a: VectorVar, b: VectorVar) -> ScalarVar { let h = self.graph.push_scalar_handle(); self.graph.ops.push(GraphOp::DotV(a.0, b.0)); ScalarVar(h) }
	fn norm_v(&mut self, v: VectorVar) -> ScalarVar { let h = self.graph.push_scalar_handle(); self.graph.ops.push(GraphOp::NormV(v.0)); ScalarVar(h) }
	fn norm_sq_v(&mut self, v: VectorVar) -> ScalarVar { let h = self.graph.push_scalar_handle(); self.graph.ops.push(GraphOp::NormSqV(v.0)); ScalarVar(h) }
	fn sum_v(&mut self, v: VectorVar) -> ScalarVar { let h = self.graph.push_scalar_handle(); self.graph.ops.push(GraphOp::SumV(v.0)); ScalarVar(h) }

	// ── Matrix ops ───────────────────────────────────────────────────

	fn add_m(&mut self, a: MatrixVar, b: MatrixVar) -> MatrixVar { let h = self.graph.push_matrix_handle(); self.graph.ops.push(GraphOp::AddM(a.0, b.0)); MatrixVar(h) }
	fn sub_m(&mut self, a: MatrixVar, b: MatrixVar) -> MatrixVar { let h = self.graph.push_matrix_handle(); self.graph.ops.push(GraphOp::SubM(a.0, b.0)); MatrixVar(h) }
	fn neg_m(&mut self, a: MatrixVar) -> MatrixVar { let h = self.graph.push_matrix_handle(); self.graph.ops.push(GraphOp::NegM(a.0)); MatrixVar(h) }
	fn scale_m(&mut self, s: ScalarVar, m: MatrixVar) -> MatrixVar { let h = self.graph.push_matrix_handle(); self.graph.ops.push(GraphOp::ScaleM(s.0, m.0)); MatrixVar(h) }
	fn mat_mul(&mut self, a: MatrixVar, b: MatrixVar) -> MatrixVar { let h = self.graph.push_matrix_handle(); self.graph.ops.push(GraphOp::MatMul(a.0, b.0)); MatrixVar(h) }
	fn mat_vec(&mut self, m: MatrixVar, v: VectorVar) -> VectorVar { let h = self.graph.push_vector_handle(); self.graph.ops.push(GraphOp::MatVec(m.0, v.0)); VectorVar(h) }
	fn transpose_m(&mut self, a: MatrixVar) -> MatrixVar { let h = self.graph.push_matrix_handle(); self.graph.ops.push(GraphOp::TransposeM(a.0)); MatrixVar(h) }
	fn trace_m(&mut self, m: MatrixVar) -> ScalarVar { let h = self.graph.push_scalar_handle(); self.graph.ops.push(GraphOp::TraceM(m.0)); ScalarVar(h) }
	fn frob_dot(&mut self, a: MatrixVar, b: MatrixVar) -> ScalarVar { let h = self.graph.push_scalar_handle(); self.graph.ops.push(GraphOp::FrobDotM(a.0, b.0)); ScalarVar(h) }

	// ── Fused ops ────────────────────────────────────────────────────

	fn linear_layer(&mut self, a: MatrixVar, x: VectorVar, b: VectorVar) -> VectorVar { let h = self.graph.push_vector_handle(); self.graph.ops.push(GraphOp::LinearLayer(a.0, x.0, b.0)); VectorVar(h) }
	fn quad_form(&mut self, x: VectorVar, a: MatrixVar) -> ScalarVar { let h = self.graph.push_scalar_handle(); self.graph.ops.push(GraphOp::QuadForm(x.0, a.0)); ScalarVar(h) }

	// ── Build problem ────────────────────────────────────────────────

	fn build_vector_problem(&mut self, _loss: ScalarVar) -> PyProblem {
		let graph = self.graph.clone();
		PyProblem::new_autodiff_vec(graph)
	}

	fn build_matrix_problem(&mut self, _loss: ScalarVar) -> PyProblem {
		let graph = self.graph.clone();
		PyProblem::new_autodiff_mat(graph)
	}
}
