use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DecodeConfig {
    pub modular_group_execution: ModularGroupExecution,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModularGroupExecution {
    #[default]
    Serial,
    RequestedThreads(usize),
}

impl DecodeConfig {
    pub fn validate(self) -> Result<Self> {
        match self.modular_group_execution {
            ModularGroupExecution::Serial => Ok(self),
            ModularGroupExecution::RequestedThreads(0) => {
                Err(Error::Unsupported("zero modular group threads"))
            }
            ModularGroupExecution::RequestedThreads(_) => Ok(self),
        }
    }
}
