use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DecodeConfig {
    pub modular_group_execution: ModularGroupExecution,
    pub region: Option<ImageRegion>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageRegion {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModularGroupExecution {
    #[default]
    Serial,
    RequestedThreads(usize),
}

impl DecodeConfig {
    pub fn validate(self) -> Result<Self> {
        if let Some(region) = self.region
            && (region.width == 0 || region.height == 0)
        {
            return Err(Error::InvalidCodestream("empty decode region"));
        }
        match self.modular_group_execution {
            ModularGroupExecution::Serial => Ok(self),
            ModularGroupExecution::RequestedThreads(0) => {
                Err(Error::Unsupported("zero modular group threads"))
            }
            ModularGroupExecution::RequestedThreads(_) => Ok(self),
        }
    }
}
