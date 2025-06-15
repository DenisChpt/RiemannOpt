# RiemannOpt Core Module - Improvements Roadmap

## Phase 1: Retractions et Transport Vectoriel

### 1.1 QR-based Retraction ✓
- [x] Implémenter `QRRetraction` dans `retraction.rs`
- [x] Ajouter support pour matrices rectangulaires
- [ ] Optimiser avec QR en place si possible (pour les manifolds spécifiques)
- [x] Ajouter tests unitaires
- [x] Documenter la complexité O(n²p)

### 1.2 Cayley Transform Retraction ✓
- [x] Implémenter `CayleyRetraction` pour matrices antisymétriques
- [x] Support pour SO(n) et variétés associées (structure de base)
- [x] Ajouter paramètre de scaling
- [x] Tests de stabilité numérique
- [ ] Benchmarks vs exponential map (pour plus tard)

### 1.3 Polar Retraction ✓
- [x] Implémenter `PolarRetraction` pour Stiefel/Grassmann
- [x] Utiliser SVD ou iteration de Newton (structure préparée)
- [ ] Gérer les cas dégénérés (pour les manifolds spécifiques)
- [x] Tests comparatifs avec QR
- [x] Documentation des trade-offs

### 1.4 Vector Transport Amélioré ✓
- [x] Implémenter `DifferentialRetraction` transport
- [x] Ajouter `SchildLadder` pour transport numérique
- [ ] Transport le long des géodésiques avec ODE (pour plus tard)
- [x] Tests de parallélisme (courbure zéro)
- [ ] Benchmarks performance (pour plus tard)

## Phase 2: Métriques Avancées

### 2.1 Christoffel Symbols Computation ✓
- [x] Ajouter méthode `compute_christoffel_symbols` dans `MetricTensor`
- [x] Implémenter différences finies pour dérivées
- [x] Support pour métriques analytiques
- [ ] Cache pour réutilisation (architecture préparée)
- [x] Tests avec métriques connues
- [x] Ajout de `geodesic_acceleration` pour calcul des géodésiques

### 2.2 Adaptive Metrics
- [ ] Créer trait `AdaptiveMetric`
- [ ] Implémenter `HessianBasedMetric`
- [ ] Support pour mise à jour pendant optimisation
- [ ] Stratégies de régularisation
- [ ] Tests de convergence

### 2.3 Affine Connections
- [ ] Créer trait `AffineConnection`
- [ ] Séparer connexion de métrique
- [ ] Implémenter connexions canoniques
- [ ] Support pour torsion non-nulle
- [ ] Documentation mathématique

## Phase 3: Algorithmes d'Optimisation

### 3.1 Trust Region Framework
- [ ] Implémenter `TrustRegionOptimizer` trait
- [ ] Créer `TrustRegionState` avec radius management
- [ ] Implémenter `CauchyPoint` solver
- [ ] Ajouter `DoglegSolver` 
- [ ] Tests sur problèmes non-convexes

### 3.2 CG-Steihaug Solver
- [ ] Implémenter `SteihhaugSolver` pour sous-problèmes
- [ ] Gérer negative curvature
- [ ] Early termination conditions
- [ ] Preconditioning support
- [ ] Integration avec trust region

### 3.3 Enhanced Line Search
- [ ] Ajouter interpolation cubique dans `StrongWolfeLineSearch`
- [ ] Implémenter `MoreThuenteLineSearch`
- [ ] Bracketing avec historique
- [ ] Safeguards numériques
- [ ] Benchmarks convergence

### 3.4 L-BFGS Improvements
- [ ] Gestion mémoire adaptative
- [ ] Scaling automatique
- [ ] Compact representation
- [ ] Preconditioning strategies
- [ ] Tests grande échelle

## Phase 4: API et Ergonomie

### 4.1 Builder Pattern
- [ ] Créer builders pour tous les optimizers
- [ ] Validation des paramètres
- [ ] Defaults intelligents
- [ ] Chainage fluide
- [ ] Documentation examples

### 4.2 Callbacks System
- [ ] Définir trait `OptimizationCallback`
- [ ] Implémenter `ProgressCallback`
- [ ] Ajouter `CheckpointCallback`
- [ ] Support pour early stopping
- [ ] Tests avec multiple callbacks

### 4.3 Structured Logging
- [ ] Intégrer `tracing` crate
- [ ] Log levels configurables
- [ ] Structured fields (iteration, value, etc)
- [ ] Performance metrics
- [ ] Optional file output

### 4.4 Monitoring Tools
- [ ] Créer `OptimizationMonitor` struct
- [ ] Collecter statistiques (gradients, steps, etc)
- [ ] Export formats (CSV, JSON)
- [ ] Visualisation helpers
- [ ] Real-time plotting hooks

## Phase 5: Performance et Optimisations

### 5.1 Batch Operations
- [ ] Ajouter `batch_retract` methods
- [ ] Parallel tangent projections
- [ ] Vectorized inner products
- [ ] Memory pooling
- [ ] Benchmarks scaling

### 5.2 SIMD Optimizations
- [ ] Identifier hot paths avec profiling
- [ ] Utiliser `packed_simd` pour ops vectorielles
- [ ] Specialized 3D/4D implementations
- [ ] Alignment optimizations
- [ ] Performance regression tests

### 5.3 Const Generics Specialization
- [ ] Templates pour dimensions communes
- [ ] Sphere<3>, Stiefel<N,P> specializations
- [ ] Stack allocation pour petites matrices
- [ ] Inline optimizations
- [ ] Size benchmarks

### 5.4 Cache-Friendly Layouts
- [ ] Analyser patterns d'accès mémoire
- [ ] Réorganiser structures pour localité
- [ ] Prefetching hints
- [ ] NUMA awareness (optional)
- [ ] Cache miss profiling

## Phase 6: Robustesse et Validation

### 6.1 Numerical Stability Tests
- [ ] Tests avec conditionnement extrême
- [ ] Matrices quasi-singulières
- [ ] Gradients très petits/grands
- [ ] Accumulation d'erreurs
- [ ] Recovery strategies

### 6.2 Comprehensive Benchmarks
- [ ] Suite de problèmes canoniques
- [ ] Comparaison avec Manopt/Pymanopt
- [ ] Profils de performance
- [ ] Scalability tests
- [ ] CI integration

### 6.3 Advanced Error Handling
- [ ] Retraction fallback chains
- [ ] Adaptive tolerance adjustment
- [ ] Diagnostic information collection
- [ ] Recovery recommendations
- [ ] Debug mode verbosity

### 6.4 Convergence Test Suite
- [ ] PCA sur Stiefel
- [ ] Moyennes sur SPD
- [ ] Low-rank matrix completion
- [ ] Robust subspace tracking
- [ ] Performance baselines

## Phase 7: Documentation et Examples

### 7.1 Implementation Guides
- [ ] "Implementing Your Own Manifold" tutorial
- [ ] "Custom Retractions" guide
- [ ] "Performance Tuning" document
- [ ] "Debugging Convergence" guide
- [ ] Code templates

### 7.2 Mathematical Documentation
- [ ] Detailed formulas for each retraction
- [ ] Complexity analysis tables
- [ ] Bibliography with key papers
- [ ] Notation conventions
- [ ] Visual diagrams

### 7.3 Example Gallery
- [ ] Principal Component Analysis
- [ ] Geometric Median
- [ ] Matrix Completion
- [ ] Tensor Decomposition
- [ ] Neural Network Training

### 7.4 Troubleshooting Resources
- [ ] Common error patterns
- [ ] Performance bottlenecks
- [ ] Numerical issues
- [ ] FAQ section
- [ ] Community examples

## Implementation Order

1. **Immediate Priority**: Phases 1.1-1.3 (Core retractions)
2. **Short Term**: Phases 3.3, 4.1-4.2 (Line search, API)
3. **Medium Term**: Phases 2.1, 3.1-3.2 (Metrics, Trust Region)
4. **Long Term**: Phases 5-7 (Performance, Documentation)

## Testing Strategy

- Run `cargo test` after each implementation
- Add unit tests for each new feature
- Integration tests for feature combinations
- Benchmark before/after optimizations
- Document performance characteristics

## Success Criteria

- All tests passing (cargo test --all)
- No compilation warnings (cargo clippy)
- Documentation for all public APIs
- Benchmarks showing improvements
- Examples demonstrating usage