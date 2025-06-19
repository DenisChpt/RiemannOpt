# Plan d'Optimisation Détaillé - RiemannOpt

## Phase 0: Préparation et Restructuration de Base

### Étape 0.1: Réorganisation des Fichiers Existants
**Objectif**: Créer la nouvelle structure de dossiers sans casser le code existant

**Actions**:
1. Créer les nouveaux dossiers dans `src/`:
   - `core/`
   - `memory/`
   - `compute/`
   - `compute/cpu/`
   - `compute/specialized/`
   - `optimization/`
   - `optimization/adaptive/`
   - `profiling/`
   - `numerical/`
   - `manifold_ops/`
   - `utils/`
   - `config/`

2. Déplacer les fichiers existants:
   - `cost_function.rs` → `core/cost_function.rs`
   - `error.rs` → `core/error.rs`
   - `manifold.rs` → `core/manifold.rs`
   - `types.rs` → `core/types.rs`
   - `simd.rs` → `compute/cpu/simd.rs`
   - `parallel.rs` → `compute/cpu/parallel.rs`
   - `parallel_thresholds.rs` → `utils/parallel_thresholds.rs`
   - `numerical_stability.rs` → `numerical/stability.rs`
   - `numerical_validation.rs` → `numerical/validation.rs`
   - `optimizer.rs` → `optimization/optimizer.rs`
   - `optimizer_state.rs` → `optimization/optimizer_state.rs`
   - `line_search.rs` → `optimization/line_search.rs`
   - `step_size.rs` → `optimization/step_size.rs`
   - `preconditioner.rs` → `optimization/preconditioner.rs`
   - `retraction.rs` → `manifold_ops/retraction.rs`
   - `tangent.rs` → `manifold_ops/tangent.rs`
   - `metric.rs` → `manifold_ops/metric.rs`
   - `fisher.rs` → `manifold_ops/fisher.rs`
   - `test_manifolds.rs` → `utils/test_manifolds.rs`
   - `test_utils.rs` → `utils/test_utils.rs`

3. Créer les fichiers `mod.rs` pour chaque nouveau module avec les ré-exports appropriés

4. Mettre à jour `lib.rs` pour utiliser la nouvelle structure de modules

5. Vérifier que tous les tests passent encore

## Phase 1: Infrastructure SIMD Avancée

### Étape 1.1: Détection des Capacités CPU
**Objectif**: Détecter dynamiquement les instructions SIMD disponibles

**Fichiers à créer**:
- `src/config/features.rs`
- `src/config/mod.rs`

**Actions**:
1. Implémenter la détection runtime des features CPU (AVX2, AVX512, FMA, NEON)
2. Créer une structure `CpuFeatures` globale accessible via lazy_static
3. Ajouter des macros pour la compilation conditionnelle selon l'architecture
4. Créer des tests unitaires pour vérifier la détection

### Étape 1.2: Refonte du Module SIMD
**Objectif**: Implémenter un dispatch runtime efficace pour les opérations vectorielles

**Fichiers à modifier**:
- `src/compute/cpu/simd.rs` (ancien `src/simd.rs`)

**Actions**:
1. Créer un trait `SimdBackend` avec les opérations de base (dot, axpy, norm, etc.)
2. Implémenter plusieurs backends: `ScalarBackend`, `Avx2Backend`, `Avx512Backend`, `NeonBackend`
3. Créer un `SimdDispatcher` qui sélectionne le backend optimal au runtime
4. Remplacer toutes les utilisations de nalgebra par les appels SIMD quand pertinent
5. Ajouter des benchmarks pour valider les gains

### Étape 1.3: Intégration SIMD dans les Opérations Critiques
**Objectif**: Utiliser SIMD dans toutes les hot paths

**Fichiers à modifier**:
- `src/core/cost_function.rs`
- `src/manifold_ops/metric.rs`
- `src/manifold_ops/tangent.rs`

**Actions**:
1. Identifier les boucles critiques via profiling
2. Remplacer les opérations nalgebra par des appels au `SimdDispatcher`
3. Ajouter des versions `_simd` des méthodes critiques
4. Créer des benchmarks comparatifs

## Phase 2: Élimination des Allocations

### Étape 2.1: Système de Pool de Mémoire
**Objectif**: Réutiliser les allocations pour éviter la pression sur l'allocateur

**Fichiers à créer**:
- `src/memory/pool.rs`
- `src/memory/mod.rs`

**Actions**:
1. Créer un `VectorPool<T, D>` thread-safe avec des pools par taille
2. Implémenter `PooledVector<T, D>` avec RAII pour le retour automatique
3. Ajouter un `MatrixPool<T, R, C>` pour les matrices
4. Créer des benchmarks de comparaison allocation vs pool

### Étape 2.2: Workspace pour les Calculs
**Objectif**: Pré-allouer tous les buffers temporaires nécessaires

**Fichiers à créer**:
- `src/memory/workspace.rs`

**Fichiers à modifier**:
- `src/optimization/optimizer_state.rs`

**Actions**:
1. Créer une structure `Workspace` contenant tous les buffers temporaires
2. Analyser chaque algorithme pour identifier ses besoins en mémoire temporaire
3. Ajouter un `workspace: Workspace` à `OptimizerState`
4. Créer des méthodes `with_workspace` pour toutes les opérations allouantes

### Étape 2.3: Refactoring des Méthodes Allouantes
**Objectif**: Éliminer toutes les allocations dans les hot paths

**Fichiers à modifier**:
- `src/core/cost_function.rs`
- `src/manifold_ops/*.rs`
- `src/optimization/*.rs`

**Actions**:
1. Pour chaque méthode allouante, créer une version `_in_place` ou `_with_buffer`
2. Modifier `gradient_fd` pour utiliser des buffers pré-alloués
3. Remplacer les `Vec::new()` et `zeros()` par des accès au workspace
4. Valider avec des tests de non-régression

## Phase 3: Parallélisation Intelligente

### Étape 3.1: Parallélisation des Calculs de Gradient
**Objectif**: Exploiter tous les cœurs pour les calculs indépendants

**Fichiers à modifier**:
- `src/compute/cpu/parallel.rs`
- `src/core/cost_function.rs`

**Actions**:
1. Ajouter `gradient_fd_parallel` utilisant rayon
2. Implémenter une heuristique basée sur la dimension pour choisir séquentiel vs parallèle
3. Créer un `ParallelConfig` pour contrôler le nombre de threads
4. Ajouter des tests de correction et des benchmarks

### Étape 3.2: Stratégie de Parallélisation Adaptative
**Objectif**: Choisir automatiquement la meilleure stratégie de parallélisation

**Fichiers à créer**:
- `src/compute/cpu/parallel_strategy.rs`

**Actions**:
1. Implémenter un système de micro-benchmarks au démarrage
2. Créer des heuristiques pour le choix du niveau de parallélisme
3. Mesurer l'overhead de parallélisation pour différentes tailles
4. Intégrer avec le système de métriques

## Phase 4: Cache Multi-Niveaux

### Étape 4.1: Infrastructure de Cache
**Objectif**: Éviter les recalculs coûteux

**Fichiers à créer**:
- `src/memory/cache.rs`

**Actions**:
1. Implémenter un cache L1 thread-local avec LRU
2. Ajouter un cache L2 concurrent avec dashmap
3. Créer des traits `Cacheable` et `CacheKey`
4. Implémenter des métriques de hit/miss ratio

### Étape 4.2: Intégration du Cache
**Objectif**: Wrapper les cost functions avec du caching transparent

**Fichiers à créer**:
- `src/core/cached_cost_function.rs`

**Actions**:
1. Créer `CachedCostFunction<F>` qui wrappe une cost function
2. Implémenter un hash stable pour les points de manifold
3. Ajouter une invalidation intelligente du cache
4. Créer des benchmarks avec différents patterns d'accès

## Phase 5: Backend Adaptatif

### Étape 5.1: Abstraction des Backends
**Objectif**: Permettre plusieurs implémentations de calcul

**Fichiers à créer**:
- `src/compute/backend.rs`
- `src/compute/mod.rs`
- `src/core/traits.rs`

**Actions**:
1. Définir le trait `ComputeBackend` avec toutes les opérations
2. Créer `CpuBackend` comme implémentation de référence
3. Implémenter `BackendSelector` pour le choix automatique
4. Ajouter des benchmarks de sélection

### Étape 5.2: Préparation Infrastructure GPU
**Objectif**: Préparer l'architecture pour l'ajout futur de GPU

**Fichiers à créer**:
- `src/compute/gpu/selector.rs`
- `src/compute/gpu/traits.rs`

**Actions**:
1. Définir les traits pour les opérations GPU
2. Créer la logique de décision CPU vs GPU
3. Implémenter la détection de GPU disponible
4. Préparer les points d'extension pour CUDA/ROCm/Metal

## Phase 6: Optimisations Spécialisées

### Étape 6.1: Optimisations Petites Dimensions
**Objectif**: Fast paths pour les cas 2D/3D courants

**Fichiers à créer**:
- `src/compute/specialized/small_dim.rs`
- `src/compute/specialized/mod.rs`

**Actions**:
1. Créer des implémentations déroulées pour dim 2, 3, 4
2. Utiliser des opérations inline et sans branches
3. Spécialiser les retractions et projections
4. Ajouter des benchmarks comparatifs

### Étape 6.2: Support Matrices Creuses
**Objectif**: Optimiser pour les problèmes avec structure creuse

**Fichiers à créer**:
- `src/compute/specialized/sparse.rs`

**Actions**:
1. Ajouter le support pour les formats CSR/CSC
2. Implémenter les opérations spécialisées (SpMV, etc.)
3. Créer des adaptateurs pour l'API existante
4. Benchmarker sur des problèmes creux typiques

## Phase 7: Profiling et Auto-Tuning

### Étape 7.1: Infrastructure de Métriques
**Objectif**: Collecter des données de performance en production

**Fichiers à créer**:
- `src/profiling/metrics.rs`
- `src/profiling/mod.rs`

**Actions**:
1. Créer un système de métriques léger et non-intrusif
2. Instrumenter les fonctions critiques
3. Implémenter l'agrégation et le reporting
4. Ajouter des hooks pour l'export des métriques

### Étape 7.2: Auto-Tuning
**Objectif**: Optimiser automatiquement les paramètres

**Fichiers à créer**:
- `src/profiling/auto_tuner.rs`
- `src/config/tuning.rs`

**Actions**:
1. Implémenter un système de benchmarking automatique
2. Créer des heuristiques d'ajustement des paramètres
3. Persister les configurations optimales
4. Ajouter des tests de convergence

## Phase 8: Intégration et Validation

### Étape 8.1: Tests de Performance
**Objectif**: Valider toutes les optimisations

**Actions dans `benches/`**:
1. Créer des benchmarks pour chaque optimisation
2. Implémenter des comparaisons avec PyManopt
3. Ajouter des tests de régression de performance
4. Créer des profils de performance pour différents cas d'usage

### Étape 8.2: Documentation et Configuration
**Objectif**: Documenter et rendre configurable

**Fichiers à créer**:
- `PERFORMANCE.md`
- `src/config/runtime.rs`

**Actions**:
1. Documenter chaque optimisation et son impact
2. Créer des features Cargo pour activer/désactiver
3. Ajouter des exemples d'utilisation optimale
4. Créer un guide de tuning

## Ordre d'Implémentation Recommandé

1. **Phase 0**: Base nécessaire pour tout le reste
2. **Phase 1**: SIMD - gain immédiat important
3. **Phase 2**: Élimination allocations - impact majeur
4. **Phase 3**: Parallélisation - gain sur grandes dimensions
5. **Phase 4**: Cache - gain pour problèmes répétitifs
6. **Phase 5**: Backend adaptatif - flexibilité future
7. **Phase 6**: Optimisations spécialisées - gains ciblés
8. **Phase 7**: Auto-tuning - optimisation continue
9. **Phase 8**: Validation finale

## Métriques de Succès

- Réduction de 10x des allocations dans les hot paths
- Amélioration de 5-20x sur les opérations vectorielles (SIMD)
- Scaling quasi-linéaire avec le nombre de cœurs
- Performance supérieure à PyManopt sur benchmarks standards
- Overhead < 5% pour les optimisations adaptatives