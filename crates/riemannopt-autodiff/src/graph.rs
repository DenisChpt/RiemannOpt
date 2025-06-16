//! Computation graph for automatic differentiation.
//!
//! This module implements a dynamic computation graph that tracks operations
//! and their dependencies for automatic differentiation.

use nalgebra::{DMatrix, DVector};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::{self, Debug};
use std::rc::Rc;

/// Unique identifier for nodes in the computation graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(usize);

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Node{}", self.0)
    }
}

/// Type alias for tensors used in the graph.
pub type Tensor = DMatrix<f64>;

/// Type alias for vectors used in the graph.
pub type Vector = DVector<f64>;

/// A variable in the computation graph.
#[derive(Debug, Clone)]
pub struct Variable {
    /// Unique identifier for this variable
    pub id: NodeId,
    /// Name of the variable (optional)
    pub name: Option<String>,
    /// Whether this variable requires gradient computation
    pub requires_grad: bool,
}

impl Variable {
    /// Creates a new variable with the given ID.
    pub fn new(id: NodeId) -> Self {
        Self {
            id,
            name: None,
            requires_grad: true,
        }
    }

    /// Sets the name of the variable.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets whether the variable requires gradient.
    pub fn with_requires_grad(mut self, requires_grad: bool) -> Self {
        self.requires_grad = requires_grad;
        self
    }
}

/// A node in the computation graph.
#[derive(Debug)]
pub struct Node {
    /// Unique identifier
    pub id: NodeId,
    /// The value computed at this node
    pub value: Option<Tensor>,
    /// The operation that produced this node
    pub op: Option<Box<dyn crate::ops::Op>>,
    /// Input nodes to this operation
    pub inputs: Vec<NodeId>,
    /// Whether this node requires gradient
    pub requires_grad: bool,
    /// Optional name for debugging
    pub name: Option<String>,
}

impl Node {
    /// Creates a new node with the given ID.
    pub fn new(id: NodeId) -> Self {
        Self {
            id,
            value: None,
            op: None,
            inputs: Vec::new(),
            requires_grad: false,
            name: None,
        }
    }

    /// Creates a new input node with a value.
    pub fn input(id: NodeId, value: Tensor) -> Self {
        Self {
            id,
            value: Some(value),
            op: None,
            inputs: Vec::new(),
            requires_grad: false,
            name: None,
        }
    }

    /// Creates a new node from an operation.
    pub fn from_op(
        id: NodeId,
        op: Box<dyn crate::ops::Op>,
        inputs: Vec<NodeId>,
        requires_grad: bool,
    ) -> Self {
        Self {
            id,
            value: None,
            op: Some(op),
            inputs,
            requires_grad,
            name: None,
        }
    }

    /// Sets the name of the node.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Checks if this node is a leaf (has no operation).
    pub fn is_leaf(&self) -> bool {
        self.op.is_none()
    }
}

/// The computation graph structure.
#[derive(Debug)]
pub struct Graph {
    /// All nodes in the graph, indexed by their ID
    nodes: RefCell<HashMap<NodeId, Rc<RefCell<Node>>>>,
    /// Counter for generating unique node IDs
    next_id: RefCell<usize>,
    /// Whether to track gradients
    track_gradients: bool,
}

impl Graph {
    /// Creates a new empty computation graph.
    pub fn new() -> Self {
        Self {
            nodes: RefCell::new(HashMap::new()),
            next_id: RefCell::new(0),
            track_gradients: true,
        }
    }

    /// Creates a new graph with gradient tracking disabled.
    pub fn no_grad() -> Self {
        Self {
            nodes: RefCell::new(HashMap::new()),
            next_id: RefCell::new(0),
            track_gradients: false,
        }
    }

    /// Generates a new unique node ID.
    fn new_node_id(&self) -> NodeId {
        let mut id = self.next_id.borrow_mut();
        let node_id = NodeId(*id);
        *id += 1;
        node_id
    }

    /// Creates a new variable (input node) in the graph.
    pub fn variable(&self, value: Tensor) -> Variable {
        let id = self.new_node_id();
        let mut node = Node::input(id, value);
        node.requires_grad = self.track_gradients;
        
        let node_rc = Rc::new(RefCell::new(node));
        self.nodes.borrow_mut().insert(id, node_rc);
        
        Variable::new(id)
    }

    /// Creates a new variable with a name.
    pub fn named_variable(&self, value: Tensor, name: impl Into<String>) -> Variable {
        let id = self.new_node_id();
        let name_str = name.into();
        let mut node = Node::input(id, value);
        node.name = Some(name_str.clone());
        node.requires_grad = self.track_gradients;
        
        let node_rc = Rc::new(RefCell::new(node));
        self.nodes.borrow_mut().insert(id, node_rc);
        
        Variable::new(id).with_name(name_str)
    }
    
    /// Creates a constant (non-differentiable) node in the graph.
    pub fn constant(&self, value: Tensor) -> NodeId {
        let id = self.new_node_id();
        let mut node = Node::input(id, value);
        node.requires_grad = false; // Constants never require gradients
        
        let node_rc = Rc::new(RefCell::new(node));
        self.nodes.borrow_mut().insert(id, node_rc);
        
        id
    }

    /// Creates a new node from an operation.
    pub fn apply_op(
        &self,
        op: Box<dyn crate::ops::Op>,
        inputs: &[NodeId],
    ) -> NodeId {
        let id = self.new_node_id();
        
        // Check if any input requires grad
        let requires_grad = if self.track_gradients {
            inputs.iter().any(|&input_id| {
                self.nodes
                    .borrow()
                    .get(&input_id)
                    .map(|n| n.borrow().requires_grad)
                    .unwrap_or(false)
            })
        } else {
            false
        };
        
        let node = Node::from_op(id, op, inputs.to_vec(), requires_grad);
        let node_rc = Rc::new(RefCell::new(node));
        self.nodes.borrow_mut().insert(id, node_rc);
        
        id
    }

    /// Gets a node by its ID.
    pub fn get_node(&self, id: NodeId) -> Option<Rc<RefCell<Node>>> {
        self.nodes.borrow().get(&id).cloned()
    }

    /// Gets the value of a node.
    pub fn get_value(&self, id: NodeId) -> Option<Tensor> {
        self.get_node(id)
            .and_then(|node| node.borrow().value.clone())
    }

    /// Sets the value of a node.
    pub fn set_value(&self, id: NodeId, value: Tensor) {
        if let Some(node) = self.get_node(id) {
            node.borrow_mut().value = Some(value);
        }
    }

    /// Performs a forward pass starting from the given node.
    pub fn forward(&self, target: NodeId) -> Option<Tensor> {
        self.forward_node(target)
    }

    /// Internal method to compute the forward pass for a node.
    fn forward_node(&self, node_id: NodeId) -> Option<Tensor> {
        let node_rc = self.get_node(node_id)?;
        
        // Check if value is already computed
        {
            let node = node_rc.borrow();
            if let Some(ref value) = node.value {
                return Some(value.clone());
            }
            
            // If this is a leaf node without a value, something is wrong
            if node.is_leaf() {
                return None;
            }
        }
        
        // Get inputs list
        let inputs = {
            let node = node_rc.borrow();
            node.inputs.clone()
        };
        
        // Compute input values
        let mut input_values = Vec::new();
        for &input_id in &inputs {
            match self.forward_node(input_id) {
                Some(val) => input_values.push(val),
                None => return None,
            }
        }
        
        // Apply the operation
        let result = {
            let node = node_rc.borrow();
            let op = node.op.as_ref()?;
            op.forward(&input_values)
        };
        
        // Store the result
        self.set_value(node_id, result.clone());
        
        Some(result)
    }

    /// Clears all computed values in the graph.
    pub fn clear_values(&self) {
        for node in self.nodes.borrow().values() {
            node.borrow_mut().value = None;
        }
    }

    /// Gets all nodes in topological order.
    pub fn topological_order(&self) -> Vec<NodeId> {
        let mut visited = HashMap::new();
        let mut order = Vec::new();
        
        for &node_id in self.nodes.borrow().keys() {
            self.visit_topological(node_id, &mut visited, &mut order);
        }
        
        // Don't reverse - the order from DFS is already correct
        order
    }

    /// Helper for topological sort using DFS.
    fn visit_topological(
        &self,
        node_id: NodeId,
        visited: &mut HashMap<NodeId, bool>,
        order: &mut Vec<NodeId>,
    ) {
        if visited.get(&node_id).copied().unwrap_or(false) {
            return;
        }
        
        visited.insert(node_id, true);
        
        if let Some(node) = self.get_node(node_id) {
            for &input_id in &node.borrow().inputs {
                self.visit_topological(input_id, visited, order);
            }
        }
        
        order.push(node_id);
    }

    /// Returns the number of nodes in the graph.
    pub fn num_nodes(&self) -> usize {
        self.nodes.borrow().len()
    }

    /// Enables gradient tracking.
    pub fn enable_grad(&mut self) {
        self.track_gradients = true;
    }

    /// Disables gradient tracking.
    pub fn disable_grad(&mut self) {
        self.track_gradients = false;
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let graph = Graph::new();
        assert_eq!(graph.num_nodes(), 0);
        assert!(graph.track_gradients);
    }

    #[test]
    fn test_variable_creation() {
        let graph = Graph::new();
        let x = graph.variable(Tensor::from_element(2, 2, 1.0));
        
        assert_eq!(graph.num_nodes(), 1);
        assert_eq!(x.id.0, 0);
        
        let value = graph.get_value(x.id).unwrap();
        assert_eq!(value.nrows(), 2);
        assert_eq!(value.ncols(), 2);
    }

    #[test]
    fn test_named_variable() {
        let graph = Graph::new();
        let x = graph.named_variable(Tensor::from_element(3, 1, 2.0), "input");
        
        assert_eq!(x.name.as_deref(), Some("input"));
    }

    #[test]
    fn test_no_grad_graph() {
        let graph = Graph::no_grad();
        assert!(!graph.track_gradients);
    }

    #[test]
    fn test_topological_order() {
        let graph = Graph::new();
        
        // Create a simple graph: x -> y -> z
        let x = graph.variable(Tensor::from_element(1, 1, 1.0));
        
        // For now, we'll test topological order with just one node
        let order = graph.topological_order();
        assert_eq!(order.len(), 1);
        assert_eq!(order[0], x.id);
    }

    #[test]
    fn test_forward_pass_leaf() {
        let graph = Graph::new();
        let x = graph.variable(Tensor::from_element(2, 2, 3.0));
        
        let result = graph.forward(x.id).unwrap();
        assert_eq!(result[(0, 0)], 3.0);
    }
}