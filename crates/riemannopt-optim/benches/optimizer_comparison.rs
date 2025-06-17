//! Benchmarks comparing different optimization algorithms
//!
//! Run with: cargo bench

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use riemannopt_optim::{SGD, SGDConfig, Adam, AdamConfig, LBFGS, LBFGSConfig};
use riemannopt_core::{
    optimizer::StoppingCriterion,
    manifold::Manifold,
    cost_function::CostFunction,
    error::Result,
};
use riemannopt_manifolds::Sphere;
use nalgebra::DVector;

/// Simple quadratic cost function for benchmarking
#[derive(Debug)]
struct QuadraticCost {
    target: DVector<f64>,
}

impl CostFunction<f64, nalgebra::Dyn> for QuadraticCost {
    fn cost(&self, x: &DVector<f64>) -> Result<f64> {
        Ok((x - &self.target).norm_squared())
    }
    
    fn gradient(&self, x: &DVector<f64>) -> Result<DVector<f64>> {
        Ok(2.0 * (x - &self.target))
    }
}

fn benchmark_sgd_variants(c: &mut Criterion) {
    let mut group = c.benchmark_group("sgd_variants");
    
    for &dim in &[10, 50, 100] {
        let sphere = Sphere::new(dim).unwrap();
        let target = sphere.random_point();
        let cost_fn = QuadraticCost { target };
        let x0 = sphere.random_point();
        
        // Benchmark vanilla SGD
        group.bench_with_input(BenchmarkId::new("vanilla", dim), &dim, |b, _| {
            b.iter(|| {
                let mut sgd = SGD::new(SGDConfig::new().with_constant_step_size(0.01));
                let stopping_criterion = StoppingCriterion::new()
                    .with_max_iterations(50)
                    .with_gradient_tolerance(1e-6);
                
                sgd.optimize(black_box(&cost_fn), black_box(&sphere), black_box(&x0), &stopping_criterion)
            });
        });
        
        // Benchmark SGD with momentum
        group.bench_with_input(BenchmarkId::new("momentum", dim), &dim, |b, _| {
            b.iter(|| {
                let mut sgd = SGD::new(
                    SGDConfig::new()
                        .with_constant_step_size(0.01)
                        .with_classical_momentum(0.9)
                );
                let stopping_criterion = StoppingCriterion::new()
                    .with_max_iterations(50)
                    .with_gradient_tolerance(1e-6);
                
                sgd.optimize(black_box(&cost_fn), black_box(&sphere), black_box(&x0), &stopping_criterion)
            });
        });
        
        // Benchmark SGD with Nesterov
        group.bench_with_input(BenchmarkId::new("nesterov", dim), &dim, |b, _| {
            b.iter(|| {
                let mut sgd = SGD::new(
                    SGDConfig::new()
                        .with_constant_step_size(0.01)
                        .with_nesterov_momentum(0.9)
                );
                let stopping_criterion = StoppingCriterion::new()
                    .with_max_iterations(50)
                    .with_gradient_tolerance(1e-6);
                
                sgd.optimize(black_box(&cost_fn), black_box(&sphere), black_box(&x0), &stopping_criterion)
            });
        });
    }
    
    group.finish();
}

fn benchmark_optimizer_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizer_comparison");
    
    let dim = 50;
    let sphere = Sphere::new(dim).unwrap();
    let target = sphere.random_point();
    let cost_fn = QuadraticCost { target };
    let x0 = sphere.random_point();
    
    // SGD
    group.bench_function("sgd", |b| {
        b.iter(|| {
            let mut sgd = SGD::new(SGDConfig::new().with_constant_step_size(0.1));
            let stopping_criterion = StoppingCriterion::new()
                .with_max_iterations(100)
                .with_gradient_tolerance(1e-8);
            
            sgd.optimize(black_box(&cost_fn), black_box(&sphere), black_box(&x0), &stopping_criterion)
        });
    });
    
    // Adam
    group.bench_function("adam", |b| {
        b.iter(|| {
            let mut adam = Adam::new(AdamConfig::new().with_learning_rate(0.1));
            let stopping_criterion = StoppingCriterion::new()
                .with_max_iterations(100)
                .with_gradient_tolerance(1e-8);
            
            adam.optimize(black_box(&cost_fn), black_box(&sphere), black_box(&x0), &stopping_criterion)
        });
    });
    
    // L-BFGS
    group.bench_function("lbfgs", |b| {
        b.iter(|| {
            let mut lbfgs = LBFGS::new(LBFGSConfig::new().with_memory_size(10));
            let stopping_criterion = StoppingCriterion::new()
                .with_max_iterations(100)
                .with_gradient_tolerance(1e-8);
            
            lbfgs.optimize(black_box(&cost_fn), black_box(&sphere), black_box(&x0), &stopping_criterion)
        });
    });
    
    group.finish();
}

fn benchmark_step_size_schedules(c: &mut Criterion) {
    use riemannopt_core::step_size::StepSizeSchedule;
    
    let mut group = c.benchmark_group("step_size_schedules");
    
    let dim = 30;
    let sphere = Sphere::new(dim).unwrap();
    let target = sphere.random_point();
    let cost_fn = QuadraticCost { target };
    let x0 = sphere.random_point();
    
    // Constant step size
    group.bench_function("constant", |b| {
        b.iter(|| {
            let mut sgd = SGD::new(
                SGDConfig::new().with_step_size(StepSizeSchedule::Constant(0.01))
            );
            let stopping_criterion = StoppingCriterion::new()
                .with_max_iterations(50)
                .with_gradient_tolerance(1e-6);
            
            sgd.optimize(black_box(&cost_fn), black_box(&sphere), black_box(&x0), &stopping_criterion)
        });
    });
    
    // Exponential decay
    group.bench_function("exponential_decay", |b| {
        b.iter(|| {
            let mut sgd = SGD::new(
                SGDConfig::new().with_step_size(
                    StepSizeSchedule::ExponentialDecay { initial: 0.1, decay_rate: 0.95 }
                )
            );
            let stopping_criterion = StoppingCriterion::new()
                .with_max_iterations(50)
                .with_gradient_tolerance(1e-6);
            
            sgd.optimize(black_box(&cost_fn), black_box(&sphere), black_box(&x0), &stopping_criterion)
        });
    });
    
    // Polynomial decay
    group.bench_function("polynomial_decay", |b| {
        b.iter(|| {
            let mut sgd = SGD::new(
                SGDConfig::new().with_step_size(
                    StepSizeSchedule::PolynomialDecay { initial: 0.1, decay_rate: 1.0, power: 0.5 }
                )
            );
            let stopping_criterion = StoppingCriterion::new()
                .with_max_iterations(50)
                .with_gradient_tolerance(1e-6);
            
            sgd.optimize(black_box(&cost_fn), black_box(&sphere), black_box(&x0), &stopping_criterion)
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_sgd_variants,
    benchmark_optimizer_comparison,
    benchmark_step_size_schedules
);
criterion_main!(benches);