//! CPU-adaptive parallelism policy for the faer linear algebra backend.
//!
//! Measures the actual thread-pool overhead and single-core throughput on the
//! running hardware, then computes per-operation thresholds that guarantee
//! parallelism only activates when it provides a net speedup.
//!
//! # Design
//!
//! Rather than hardcoded FLOP thresholds (which assume a specific CPU), we:
//!
//! 1. **Detect** hardware properties at init: core count, SIMD width, cache
//!    sizes via CPUID and OS queries.
//!
//! 2. **Measure** two key quantities with a ~1ms micro-calibration:
//!    - `fork_join_overhead_ns`: wall-clock cost of one Rayon dispatch
//!    - `seq_gflops`: sustained single-core dgemm throughput
//!
//! 3. **Derive** per-operation thresholds from the invariant:
//!    ```text
//!    threshold_flops = fork_join_overhead_ns × seq_gflops × safety_margin / efficiency
//!    ```
//!    where `efficiency` is a per-operation constant (0.8 for GEMM,
//!    0.45 for QR, etc.) reflecting how well each algorithm scales.
//!
//! The result: on a fast AVX-512 machine with low Rayon overhead, thresholds
//! are higher (single core is fast, need more work to justify fork/join).
//! On a many-core machine with slow single-core, thresholds are lower.
//!
//! # References
//!
//! - Intel MKL Developer Reference: "Improving Performance with Threading"
//! - OpenBLAS `driver/others/parameter.c` (per-architecture threshold tables)
//! - Eigen `src/Core/products/Parallelizer.h` (cost-model dispatch)
//! - BLIS framework `frame/thread/bli_thread.c` (adaptive thread control)

use faer::Par;
use std::sync::OnceLock;

use std::sync::atomic::{AtomicBool, Ordering};

// ═══════════════════════════════════════════════════════════════════════════
//  Hardware detection
// ═══════════════════════════════════════════════════════════════════════════

/// Detected CPU capabilities, populated once at init.
#[derive(Debug, Clone)]
pub struct CpuInfo {
	/// Number of physical cores (not hyperthreads).
	pub physical_cores: usize,
	/// Number of logical cores (including hyperthreads).
	pub logical_cores: usize,
	/// Maximum SIMD register width in bits (128=SSE, 256=AVX2, 512=AVX-512).
	pub simd_width_bits: u32,
	/// L1 data cache size per core, in bytes.
	pub l1d_cache_bytes: usize,
	/// L2 cache size per core, in bytes.
	pub l2_cache_bytes: usize,
	/// L3 (last-level) cache total, in bytes. 0 if unknown.
	pub l3_cache_bytes: usize,
}

impl CpuInfo {
	/// Detect hardware properties from CPUID and OS queries.
	pub fn detect() -> Self {
		let logical_cores = std::thread::available_parallelism()
			.map(|n| n.get())
			.unwrap_or(1);

		let physical_cores = Self::detect_physical_cores(logical_cores);
		let simd_width_bits = Self::detect_simd_width();
		let (l1d, l2, l3) = Self::detect_cache_sizes();

		CpuInfo {
			physical_cores,
			logical_cores,
			simd_width_bits,
			l1d_cache_bytes: l1d,
			l2_cache_bytes: l2,
			l3_cache_bytes: l3,
		}
	}

	fn detect_physical_cores(logical: usize) -> usize {
		#[cfg(target_os = "linux")]
		{
			if let Ok(content) = std::fs::read_to_string(
				"/sys/devices/system/cpu/cpu0/topology/thread_siblings_list",
			) {
				let siblings = content.trim().split(&['-', ','][..]).count();
				if siblings > 0 {
					return logical / siblings;
				}
			}
		}
		if logical >= 4 {
			logical / 2
		} else {
			logical
		}
	}

	fn detect_simd_width() -> u32 {
		#[cfg(target_arch = "x86_64")]
		{
			if std::arch::is_x86_feature_detected!("avx512f") {
				return 512;
			}
			if std::arch::is_x86_feature_detected!("avx2") {
				return 256;
			}
			if std::arch::is_x86_feature_detected!("sse2") {
				return 128;
			}
		}
		#[cfg(target_arch = "aarch64")]
		{
			return 128; // NEON
		}
		#[allow(unreachable_code)]
		128
	}

	fn detect_cache_sizes() -> (usize, usize, usize) {
		#[cfg(target_arch = "x86_64")]
		{
			if let Some(sizes) = Self::cache_sizes_from_cpuid() {
				return sizes;
			}
		}
		#[cfg(target_os = "linux")]
		{
			if let Some(sizes) = Self::cache_sizes_from_sysfs() {
				return sizes;
			}
		}
		// Conservative fallback.
		(32 * 1024, 256 * 1024, 8 * 1024 * 1024)
	}

	#[cfg(target_arch = "x86_64")]
	fn cache_sizes_from_cpuid() -> Option<(usize, usize, usize)> {
		use core::arch::x86_64::__cpuid_count;

		let mut l1d: usize = 0;
		let mut l2: usize = 0;
		let mut l3: usize = 0;

		// Try Intel leaf 0x04, then AMD leaf 0x8000001D.
		for &leaf in &[0x04u32, 0x8000001D] {
			for subleaf in 0..16u32 {
				let r = __cpuid_count(leaf, subleaf);
				let cache_type = r.eax & 0x1F;
				if cache_type == 0 {
					break;
				}
				let level = (r.eax >> 5) & 0x07;
				let line_size = ((r.ebx) & 0xFFF) + 1;
				let partitions = ((r.ebx >> 12) & 0x3FF) + 1;
				let ways = ((r.ebx >> 22) & 0x3FF) + 1;
				let sets = r.ecx + 1;
				let size = ways as usize * partitions as usize * line_size as usize * sets as usize;

				match (level, cache_type) {
					(1, 1) | (1, 3) => l1d = l1d.max(size),
					(2, _) => l2 = l2.max(size),
					(3, _) => l3 = l3.max(size),
					_ => {}
				}
			}
			if l1d > 0 {
				break;
			}
		}

		if l1d > 0 && l2 > 0 {
			Some((l1d, l2, l3))
		} else {
			None
		}
	}

	#[cfg(target_os = "linux")]
	fn cache_sizes_from_sysfs() -> Option<(usize, usize, usize)> {
		let mut l1d = 0usize;
		let mut l2 = 0usize;
		let mut l3 = 0usize;

		for index in 0..8 {
			let base = format!("/sys/devices/system/cpu/cpu0/cache/index{index}");
			let level: usize = std::fs::read_to_string(format!("{base}/level"))
				.ok()
				.and_then(|s| s.trim().parse().ok())
				.unwrap_or(0);
			let ctype = std::fs::read_to_string(format!("{base}/type"))
				.unwrap_or_default()
				.trim()
				.to_lowercase();
			let size = std::fs::read_to_string(format!("{base}/size"))
				.ok()
				.and_then(|s| {
					let s = s.trim();
					if let Some(k) = s.strip_suffix('K') {
						k.parse::<usize>().ok().map(|v| v * 1024)
					} else if let Some(m) = s.strip_suffix('M') {
						m.parse::<usize>().ok().map(|v| v * 1024 * 1024)
					} else {
						s.parse().ok()
					}
				})
				.unwrap_or(0);

			match level {
				1 if ctype == "data" || ctype == "unified" => l1d = l1d.max(size),
				2 => l2 = l2.max(size),
				3 => l3 = l3.max(size),
				_ => {}
			}
		}

		if l1d > 0 && l2 > 0 {
			Some((l1d, l2, l3))
		} else {
			None
		}
	}

	/// Estimated peak single-core dgemm GFLOP/s from SIMD width alone.
	/// Used only as fallback when calibration is skipped.
	pub fn estimated_peak_gflops(&self) -> f64 {
		let doubles_per_simd = self.simd_width_bits as f64 / 64.0;
		let fma_factor = 2.0;
		let assumed_ghz = 3.0;
		// Sustained ≈ 70% of peak for in-cache GEMM.
		doubles_per_simd * fma_factor * assumed_ghz * 0.7
	}
}

// ═══════════════════════════════════════════════════════════════════════════
//  Runtime calibration
// ═══════════════════════════════════════════════════════════════════════════

/// Measured performance characteristics of the current machine.
#[derive(Debug, Clone)]
pub struct Calibration {
	/// Cost of one Rayon fork+join cycle, in nanoseconds.
	pub fork_join_overhead_ns: u64,
	/// Sustained single-core dgemm throughput, in GFLOP/s.
	pub seq_gflops: f64,
}

impl Calibration {
	/// Run micro-benchmarks (~2ms total) to measure actual overhead and throughput.
	pub fn measure() -> Self {
		Calibration {
			fork_join_overhead_ns: Self::measure_rayon_overhead(),
			seq_gflops: Self::measure_seq_throughput(),
		}
	}

	/// Estimate from CPU features without benchmarking (~0 cost).
	pub fn estimate(cpu: &CpuInfo) -> Self {
		let base_ns: u64 = if cfg!(target_os = "linux") {
			1_500
		} else if cfg!(target_os = "macos") {
			4_000
		} else {
			3_000
		};
		Calibration {
			fork_join_overhead_ns: base_ns + (cpu.physical_cores as u64) * 50,
			seq_gflops: cpu.estimated_peak_gflops(),
		}
	}

	fn measure_rayon_overhead() -> u64 {
		use std::time::Instant;

		// Warm up.
		for _ in 0..10 {
			rayon::scope(|_| std::hint::black_box(()));
		}

		// Collect samples.
		let n = 80;
		let mut samples = Vec::with_capacity(n);
		for _ in 0..n {
			let t = Instant::now();
			rayon::scope(|_| std::hint::black_box(()));
			samples.push(t.elapsed().as_nanos() as u64);
		}
		samples.sort_unstable();

		// Use 10th percentile: robust against occasional OS jitter
		// but not as noisy as the absolute minimum.
		samples[n / 10]
	}

	fn measure_seq_throughput() -> f64 {
		use std::time::Instant;

		// 64×64 fits in L1 (32KB), giving peak in-cache throughput.
		let n = 64usize;
		let a = faer::Mat::<f64>::from_fn(n, n, |i, j| ((i * 31 + j * 17) % 97) as f64 * 0.01);
		let b = faer::Mat::<f64>::from_fn(n, n, |i, j| ((i * 13 + j * 23) % 89) as f64 * 0.01);
		let mut c = faer::Mat::<f64>::zeros(n, n);

		// Warmup
		for _ in 0..20 {
			faer::linalg::matmul::matmul(
				c.as_mut(),
				faer::Accum::Replace,
				a.as_ref(),
				b.as_ref(),
				1.0,
				Par::Seq,
			);
			std::hint::black_box(&c);
		}

		// Measure
		let iters = 200;
		let t = Instant::now();
		for _ in 0..iters {
			faer::linalg::matmul::matmul(
				c.as_mut(),
				faer::Accum::Replace,
				a.as_ref(),
				b.as_ref(),
				1.0,
				Par::Seq,
			);
			std::hint::black_box(&c);
		}
		let ns = t.elapsed().as_nanos() as f64;

		let flops_per_call = (2 * n * n * n) as f64;
		let gflops = (flops_per_call * iters as f64) / ns;
		gflops.clamp(1.0, 200.0)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
//  Per-operation efficiency model
// ═══════════════════════════════════════════════════════════════════════════
//
//  Each operation has a "parallel efficiency" ∈ (0, 1] modelling what
//  fraction of theoretical speedup is actually achieved, and a
//  "min_parallel_dim" below which the splitting granularity is too small.
//
//  Sources: MKL Optimization Reference, Eigen Parallelizer.h,
//  OpenBLAS parameter.c, and empirical measurements.

#[derive(Debug, Clone, Copy)]
struct OpModel {
	/// Fraction of ideal speedup achieved (Amdahl + overhead).
	efficiency: f64,
	/// Minimum dimension of the axis that gets split.
	min_parallel_dim: usize,
}

const GEMM_MODEL: OpModel = OpModel {
	efficiency: 0.80,
	min_parallel_dim: 64,
};
const GEMV_MODEL: OpModel = OpModel {
	efficiency: 0.30,
	min_parallel_dim: 4096,
};
const QR_MODEL: OpModel = OpModel {
	efficiency: 0.45,
	min_parallel_dim: 64,
};
const HOUSEHOLDER_MODEL: OpModel = OpModel {
	efficiency: 0.50,
	min_parallel_dim: 64,
};
const SVD_MODEL: OpModel = OpModel {
	efficiency: 0.35,
	min_parallel_dim: 64,
};
const EIGEN_MODEL: OpModel = OpModel {
	efficiency: 0.30,
	min_parallel_dim: 96,
};
const CHOLESKY_MODEL: OpModel = OpModel {
	efficiency: 0.40,
	min_parallel_dim: 96,
};
const BLAS1_MODEL: OpModel = OpModel {
	efficiency: 0.20,
	min_parallel_dim: 100_000,
};

/// How many multiples of fork/join overhead the parallel work must exceed.
///
/// Derivation: for P cores and efficiency e, parallelism helps when
/// T_seq > overhead × eP/(eP−1). For P=8 e=0.7 this is 1.22×.
/// We use 5× as safety margin for cache warming, load imbalance,
/// and measurement variance. This is consistent with Eigen (which
/// uses an implicit ~4-6× margin) and OpenBLAS (3-5×).
const SAFETY_MARGIN: f64 = 5.0;

// ═══════════════════════════════════════════════════════════════════════════
//  Computed thresholds
// ═══════════════════════════════════════════════════════════════════════════

/// All thresholds, in FLOPs, computed from hardware measurements.
#[derive(Debug, Clone)]
pub struct Thresholds {
	pub gemm: usize,
	pub gemm_t: usize,
	pub gemv: usize,
	pub qr: usize,
	pub qr_min_cols: usize,
	pub householder_apply: usize,
	pub svd: usize,
	pub eigen: usize,
	pub cholesky: usize,
	pub blas1: usize,
}

impl Thresholds {
	/// Derive thresholds from calibration + CPU info.
	///
	/// Core formula:
	/// ```text
	/// base = overhead_ns × seq_gflops × SAFETY_MARGIN
	/// threshold(op) = base / op.efficiency
	/// ```
	///
	/// A fast CPU (high seq_gflops) raises all thresholds, because the
	/// single core can handle more work before fork/join pays off.
	/// High Rayon overhead also raises thresholds (more to amortise).
	pub fn compute(cal: &Calibration, cpu: &CpuInfo) -> Self {
		// base unit: the number of FLOPs that take SAFETY_MARGIN × overhead
		// to execute sequentially.
		let base = cal.fork_join_overhead_ns as f64 * cal.seq_gflops * SAFETY_MARGIN;

		let threshold =
			|model: &OpModel| -> usize { (base / model.efficiency).max(1000.0) as usize };

		// More cores → need wider min dimension so each core gets ≥1 block.
		let core_scale = (cpu.physical_cores as f64 / 4.0).max(1.0);

		Thresholds {
			gemm: threshold(&GEMM_MODEL),
			gemm_t: (threshold(&GEMM_MODEL) as f64 * 1.15) as usize,
			gemv: threshold(&GEMV_MODEL),
			qr: threshold(&QR_MODEL),
			qr_min_cols: (QR_MODEL.min_parallel_dim as f64 * core_scale.sqrt()) as usize,
			householder_apply: threshold(&HOUSEHOLDER_MODEL),
			svd: threshold(&SVD_MODEL),
			eigen: threshold(&EIGEN_MODEL),
			cholesky: threshold(&CHOLESKY_MODEL),
			blas1: threshold(&BLAS1_MODEL),
		}
	}
}

// ═══════════════════════════════════════════════════════════════════════════
//  Policy (public API)
// ═══════════════════════════════════════════════════════════════════════════

static GLOBAL_POLICY: OnceLock<Policy> = OnceLock::new();
static FORCE_SEQUENTIAL: AtomicBool = AtomicBool::new(false);

/// CPU-adaptive parallelism policy.
///
/// # Examples
///
/// ```rust
/// // At startup (once):
/// let policy = Policy::init_with_calibration();
/// println!("{}", policy.summary());
///
/// // In hot code (inline, zero-cost after init):
/// let par = Policy::global().gemm(m, n, k);
/// faer::linalg::matmul::matmul(c, accum, a, b, alpha, par);
/// ```
#[derive(Debug, Clone)]
pub struct Policy {
	pub cpu: CpuInfo,
	pub calibration: Calibration,
	pub thresholds: Thresholds,
}

impl Policy {
	/// Init with CPU detection only (no benchmark, ~0 cost).
	pub fn init() -> &'static Policy {
		GLOBAL_POLICY.get_or_init(Self::from_estimate)
	}

	/// Init with runtime calibration (~2ms cost, more accurate).
	pub fn init_with_calibration() -> &'static Policy {
		GLOBAL_POLICY.get_or_init(Self::from_calibration)
	}

	/// Lazy access: auto-inits with estimate if not already initialised.
	#[inline(always)]
	pub fn get_or_init() -> &'static Policy {
		GLOBAL_POLICY.get_or_init(Self::from_estimate)
	}

	/// Force all operations to run sequentially, regardless of size.
	/// Takes effect immediately on all threads.
	pub fn disable_parallelism() {
		FORCE_SEQUENTIAL.store(true, Ordering::Relaxed);
	}

	/// Re-enable adaptive parallelism (the default).
	pub fn enable_parallelism() {
		FORCE_SEQUENTIAL.store(false, Ordering::Relaxed);
	}

	/// Returns `true` if parallelism is currently force-disabled.
	pub fn is_parallelism_disabled() -> bool {
		FORCE_SEQUENTIAL.load(Ordering::Relaxed)
	}

	/// Access the global policy. Panics if not initialised.
	#[inline(always)]
	pub fn global() -> &'static Policy {
		GLOBAL_POLICY.get().expect(
			"parallel_policy::Policy not initialised. \
             Call Policy::init() or Policy::init_with_calibration() at startup.",
		)
	}

	fn from_estimate() -> Policy {
		let cpu = CpuInfo::detect();
		let cal = Calibration::estimate(&cpu);
		let thresholds = Thresholds::compute(&cal, &cpu);
		Policy {
			cpu,
			calibration: cal,
			thresholds,
		}
	}

	fn from_calibration() -> Policy {
		let cpu = CpuInfo::detect();
		let cal = Calibration::measure();
		let thresholds = Thresholds::compute(&cal, &cpu);
		Policy {
			cpu,
			calibration: cal,
			thresholds,
		}
	}

	// ═══════════════════════════════════════════════════════════════════
	//  Dispatch methods — all #[inline(always)]
	// ═══════════════════════════════════════════════════════════════════

	/// GEMM: C(m×n) = α · A(m×k) · B(k×n) + β · C
	#[inline(always)]
	pub fn gemm(&self, m: usize, n: usize, k: usize) -> Par {
		if m.max(n) < GEMM_MODEL.min_parallel_dim {
			return Par::Seq;
		}
		dispatch(2 * m * n * k, self.thresholds.gemm)
	}

	/// GEMM with one transposed operand.
	#[inline(always)]
	pub fn gemm_t(&self, m: usize, n: usize, k: usize) -> Par {
		if m.max(n) < GEMM_MODEL.min_parallel_dim {
			return Par::Seq;
		}
		dispatch(2 * m * n * k, self.thresholds.gemm_t)
	}

	/// GEMV: y(m) = α · A(m×n) · x(n) + β · y
	#[inline(always)]
	pub fn gemv(&self, m: usize, n: usize) -> Par {
		if m < GEMV_MODEL.min_parallel_dim {
			return Par::Seq;
		}
		dispatch(2 * m * n, self.thresholds.gemv)
	}

	/// Dot product, axpy, nrm2, scal on vector of length n.
	#[inline(always)]
	pub fn blas1(&self, n: usize) -> Par {
		if n < BLAS1_MODEL.min_parallel_dim {
			return Par::Seq;
		}
		dispatch(2 * n, self.thresholds.blas1)
	}

	/// QR factorisation of A(m×n).
	#[inline(always)]
	pub fn qr(&self, m: usize, n: usize) -> Par {
		let min_dim = m.min(n);
		if min_dim < self.thresholds.qr_min_cols {
			return Par::Seq;
		}
		let max_dim = m.max(n);
		let flops = 2 * max_dim * min_dim * min_dim - 2 * min_dim * min_dim * min_dim / 3;
		dispatch(flops, self.thresholds.qr)
	}

	/// Householder apply (m=height, n=target cols, k=reflectors).
	#[inline(always)]
	pub fn householder_apply(&self, m: usize, n: usize, k: usize) -> Par {
		if n < HOUSEHOLDER_MODEL.min_parallel_dim {
			return Par::Seq;
		}
		dispatch(4 * m * n * k, self.thresholds.householder_apply)
	}

	/// Thin SVD of A(m×n).
	#[inline(always)]
	pub fn svd(&self, m: usize, n: usize) -> Par {
		let min_dim = m.min(n);
		if min_dim < SVD_MODEL.min_parallel_dim {
			return Par::Seq;
		}
		let max_dim = m.max(n);
		let flops = 4 * max_dim * min_dim * min_dim + 8 * min_dim * min_dim * min_dim;
		dispatch(flops, self.thresholds.svd)
	}

	/// Symmetric eigendecomposition of A(n×n).
	#[inline(always)]
	pub fn eigen(&self, n: usize) -> Par {
		if n < EIGEN_MODEL.min_parallel_dim {
			return Par::Seq;
		}
		dispatch(4 * n * n * n / 3, self.thresholds.eigen)
	}

	/// Cholesky factorisation of A(n×n).
	#[inline(always)]
	pub fn cholesky(&self, n: usize) -> Par {
		if n < CHOLESKY_MODEL.min_parallel_dim {
			return Par::Seq;
		}
		dispatch(n * n * n / 3, self.thresholds.cholesky)
	}

	// ═══════════════════════════════════════════════════════════════════
	//  Diagnostics
	// ═══════════════════════════════════════════════════════════════════

	/// Human-readable summary of detected hardware and computed thresholds.
	pub fn summary(&self) -> String {
		let t = &self.thresholds;
		let gemm_break = approx_cube_root(t.gemm / 2);
		format!(
			"riemannopt::parallel_policy::Policy {{\n\
             \x20 Hardware:\n\
             \x20   cores: {} physical / {} logical\n\
             \x20   SIMD:  {}-bit\n\
             \x20   L1d:   {} KB, L2: {} KB, L3: {} MB\n\
             \x20 Calibration:\n\
             \x20   rayon fork/join overhead: {} ns\n\
             \x20   sequential dgemm:        {:.1} GFLOP/s\n\
             \x20 Thresholds (FLOPs → break-even size):\n\
             \x20   gemm:        {:>9}  (~{n}×{n}×{n} square)\n\
             \x20   gemm_t:      {:>9}\n\
             \x20   gemv:        {:>9}\n\
             \x20   qr:          {:>9}  (min_cols ≥ {})\n\
             \x20   householder: {:>9}\n\
             \x20   svd:         {:>9}\n\
             \x20   eigen:       {:>9}  (min n ≥ {})\n\
             \x20   cholesky:    {:>9}  (min n ≥ {})\n\
             \x20   blas1:       {:>9}  (min len ≥ {})\n\
             }}",
			self.cpu.physical_cores,
			self.cpu.logical_cores,
			self.cpu.simd_width_bits,
			self.cpu.l1d_cache_bytes / 1024,
			self.cpu.l2_cache_bytes / 1024,
			self.cpu.l3_cache_bytes / 1024 / 1024,
			self.calibration.fork_join_overhead_ns,
			self.calibration.seq_gflops,
			t.gemm,
			t.gemm_t,
			t.gemv,
			t.qr,
			t.qr_min_cols,
			t.householder_apply,
			t.svd,
			t.eigen,
			EIGEN_MODEL.min_parallel_dim,
			t.cholesky,
			CHOLESKY_MODEL.min_parallel_dim,
			t.blas1,
			BLAS1_MODEL.min_parallel_dim,
			n = gemm_break,
		)
	}

	/// Print a decision table for common Riemannian optimisation sizes.
	pub fn decision_table(&self) -> String {
		let cases: &[(&str, &str, usize, usize)] = &[
			("GEMM project", "Gr(50,10)", 50, 10),
			("GEMM project", "Gr(100,20)", 100, 20),
			("GEMM project", "Gr(200,20)", 200, 20),
			("GEMM project", "Gr(500,10)", 500, 10),
			("GEMM project", "Gr(500,50)", 500, 50),
			("GEMM project", "Gr(1000,100)", 1000, 100),
			("QR retract", "Gr(50,10)", 50, 10),
			("QR retract", "Gr(100,20)", 100, 20),
			("QR retract", "Gr(200,20)", 200, 20),
			("QR retract", "Gr(500,10)", 500, 10),
			("QR retract", "Gr(500,50)", 500, 50),
			("QR retract", "Gr(1000,100)", 1000, 100),
			("dot/axpy", "S(50)", 50, 1),
			("dot/axpy", "S(1000)", 1000, 1),
			("dot/axpy", "S(100000)", 100000, 1),
		];

		let mut lines = vec![format!(
			"{:<16} {:<14} {:>10} {:>10}  {}",
			"Operation", "Manifold", "Est FLOPs", "Threshold", "Decision"
		)];
		lines.push("─".repeat(70));

		for &(op, manifold, n, p) in cases {
			let (flops, threshold, decision) = if op.starts_with("GEMM") {
				// project_tangent: 2 GEMMs of total 4np²
				let f = 4 * n * p * p;
				let par = self.gemm(n, p, p);
				(f, self.thresholds.gemm, par)
			} else if op.starts_with("QR") {
				let min_dim = n.min(p);
				let max_dim = n.max(p);
				let f = 2 * max_dim * min_dim * min_dim - 2 * min_dim * min_dim * min_dim / 3;
				let par = self.qr(n, p);
				(f, self.thresholds.qr, par)
			} else {
				let f = 2 * n;
				let par = self.blas1(n);
				(f, self.thresholds.blas1, par)
			};

			let dec = if matches!(decision, Par::Seq) {
				"→ Seq"
			} else {
				"→ Par"
			};
			lines.push(format!(
				"{:<16} {:<14} {:>10} {:>10}  {}",
				op, manifold, flops, threshold, dec
			));
		}
		lines.join("\n")
	}
}

// ═══════════════════════════════════════════════════════════════════════════
//  Internals
// ═══════════════════════════════════════════════════════════════════════════

#[inline(always)]
fn dispatch(estimated_flops: usize, threshold: usize) -> Par {
	if FORCE_SEQUENTIAL.load(Ordering::Relaxed) ||estimated_flops <= threshold {
		Par::Seq
	} else {
		faer::get_global_parallelism()
	}
}

fn approx_cube_root(n: usize) -> usize {
	(n as f64).cbrt().round() as usize
}

// ═══════════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
	use super::*;

	fn make_policy() -> Policy {
		Policy::from_estimate()
	}

	#[test]
	fn cpu_detection_sane() {
		let cpu = CpuInfo::detect();
		println!("{:?}", cpu);
		assert!(cpu.physical_cores >= 1);
		assert!(cpu.logical_cores >= cpu.physical_cores);
		assert!([128, 256, 512].contains(&cpu.simd_width_bits));
		assert!(cpu.l1d_cache_bytes >= 8 * 1024);
		assert!(cpu.l2_cache_bytes > cpu.l1d_cache_bytes);
	}

	#[test]
	fn threshold_ordering() {
		let pol = make_policy();
		let t = &pol.thresholds;
		// Less efficient ops need more FLOPs to justify parallelism.
		assert!(t.qr > t.gemm, "QR should need more FLOPs than GEMM");
		assert!(t.svd > t.qr, "SVD should need more FLOPs than QR");
		assert!(t.blas1 > t.gemm, "BLAS-1 should need more FLOPs than GEMM");
	}

	#[test]
	fn benchmark_sizes_are_sequential() {
		let pol = make_policy();
		// All Grassmann benchmark sizes from the paper.
		for &(n, p) in &[(50, 10), (100, 20), (200, 20), (500, 10)] {
			let qr_par = pol.qr(n, p);
			assert!(
				matches!(qr_par, Par::Seq),
				"QR({n},{p}) should be Seq, threshold={}",
				pol.thresholds.qr
			);
		}
	}

	#[test]
	fn large_problems_go_parallel() {
		let pol = make_policy();
		// 1000×100 should go parallel for QR
		let min_dim = 100usize;
		let max_dim = 1000usize;
		let flops = 2 * max_dim * min_dim * min_dim - 2 * min_dim * min_dim * min_dim / 3;
		assert!(
			min_dim >= pol.thresholds.qr_min_cols,
			"min_dim={min_dim} should exceed qr_min_cols={}",
			pol.thresholds.qr_min_cols
		);
		assert!(
			flops >= pol.thresholds.qr,
			"QR(1000,100) flops={flops} should exceed threshold={}",
			pol.thresholds.qr
		);
	}

	#[test]
	fn sphere_always_seq() {
		let pol = make_policy();
		for n in [50, 500, 1000, 10_000] {
			assert!(matches!(pol.blas1(n), Par::Seq));
		}
	}

	#[test]
	fn prints_summary_and_table() {
		let pol = make_policy();
		println!("{}", pol.summary());
		println!();
		println!("{}", pol.decision_table());
	}

	#[test]
	#[ignore] // ~2ms, run with: cargo test -- --ignored
	fn calibration_sane() {
		let cal = Calibration::measure();
		println!(
			"overhead={} ns, throughput={:.1} GFLOP/s",
			cal.fork_join_overhead_ns, cal.seq_gflops
		);
		assert!((100..100_000).contains(&cal.fork_join_overhead_ns));
		assert!((1.0..200.0).contains(&cal.seq_gflops));
	}
}
