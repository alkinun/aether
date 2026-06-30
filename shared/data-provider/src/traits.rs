use anyhow::{bail, Result};
use psyche_core::{BatchId, TokenSize};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct TokenizedData {
    pub input_ids: Vec<i32>,
    pub labels: Option<Vec<i32>>,
    pub position_ids: Option<Vec<i32>>,
    pub sequence_lengths: Option<Vec<i32>>,
}

impl TokenizedData {
    pub fn from_input_ids(input_ids: Vec<i32>) -> Self {
        Self {
            input_ids,
            labels: None,
            position_ids: None,
            sequence_lengths: None,
        }
    }

    pub fn new(
        input_ids: Vec<i32>,
        labels: Option<Vec<i32>>,
        position_ids: Option<Vec<i32>>,
        sequence_lengths: Option<Vec<i32>>,
    ) -> Self {
        Self {
            input_ids,
            labels,
            position_ids,
            sequence_lengths,
        }
    }

    pub fn empty() -> Self {
        Self {
            input_ids: vec![],
            labels: None,
            position_ids: None,
            sequence_lengths: None,
        }
    }
}

pub trait TokenizedDataProvider {
    fn get_samples(
        &mut self,
        data_ids: BatchId,
    ) -> impl std::future::Future<Output = Result<Vec<TokenizedData>>> + Send;
}

pub trait LengthKnownDataProvider {
    fn num_sequences(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.num_sequences() == 0
    }
}

pub(crate) fn bytes_to_tokens(data: &[u8], token_size: TokenSize) -> Result<Vec<i32>> {
    let token_len = usize::from(token_size);
    if data.len() % token_len != 0 {
        bail!(
            "token data length {} is not divisible by token size {}",
            data.len(),
            token_len
        );
    }

    Ok(data
        .chunks_exact(token_len)
        .map(|chunk| match token_size {
            TokenSize::TwoBytes => u16::from_le_bytes([chunk[0], chunk[1]]) as i32,
            TokenSize::FourBytes => {
                u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as i32
            }
        })
        .collect())
}
