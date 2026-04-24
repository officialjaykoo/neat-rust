use std::fmt;

pub type NodeKey = i64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConnectionKey {
    pub input: NodeKey,
    pub output: NodeKey,
}

impl ConnectionKey {
    pub const fn new(input: NodeKey, output: NodeKey) -> Self {
        Self { input, output }
    }
}

impl fmt::Display for ConnectionKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.input, self.output)
    }
}
