use crate::{traits::TokenizedDataProvider, LengthKnownDataProvider, TokenizedData};
use anyhow::{bail, Result};
use psyche_core::BatchId;

pub struct DummyDataProvider {
    seq_len: usize,
    num_sequences: u64,
}

impl DummyDataProvider {
    pub fn new(
        _token_size_in_bytes: psyche_core::TokenSize,
        num_tokens_per_sequence: usize, // num tokens per sequence
        num_sequences: u64,
    ) -> Self {
        Self {
            seq_len: num_tokens_per_sequence,
            num_sequences,
        }
    }

    fn internal_get_samples(&self, num_samples: usize) -> Result<Vec<TokenizedData>> {
        let mut ret: Vec<_> = Vec::new();
        for _ in 0..num_samples {
            ret.push(TokenizedData::from_input_ids(vec![0; self.seq_len]));
        }
        Ok(ret)
    }
}

impl TokenizedDataProvider for DummyDataProvider {
    async fn get_samples(&mut self, data_ids: BatchId) -> Result<Vec<TokenizedData>> {
        for id in data_ids.iter() {
            if id > self.num_sequences {
                bail!("id {id} > self.num_sequences {}", self.num_sequences)
            }
        }
        self.internal_get_samples(data_ids.len())
    }
}

impl LengthKnownDataProvider for DummyDataProvider {
    fn num_sequences(&self) -> usize {
        self.num_sequences as usize
    }
}
