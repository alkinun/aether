use crate::{
    http::HttpDataProvider, DummyDataProvider, LocalDataProvider, PreprocessedDataProvider,
    TokenizedData, TokenizedDataProvider, WeightedDataProvider,
};

#[cfg(feature = "remote")]
use crate::DataProviderTcpClient;

use psyche_core::BatchId;

pub enum DataProvider {
    Http(HttpDataProvider),
    #[cfg(feature = "remote")]
    Server(DataProviderTcpClient),
    Dummy(DummyDataProvider),
    WeightedHttp(WeightedDataProvider<HttpDataProvider>),
    Local(LocalDataProvider),
    Preprocessed(PreprocessedDataProvider),
}

impl TokenizedDataProvider for DataProvider {
    async fn get_samples(&mut self, data_ids: BatchId) -> anyhow::Result<Vec<TokenizedData>> {
        match self {
            DataProvider::Http(provider) => provider.get_samples(data_ids).await,
            #[cfg(feature = "remote")]
            DataProvider::Server(provider) => provider.get_samples(data_ids).await,
            DataProvider::Dummy(provider) => provider.get_samples(data_ids).await,
            DataProvider::WeightedHttp(provider) => provider.get_samples(data_ids).await,
            DataProvider::Local(provider) => provider.get_samples(data_ids).await,
            DataProvider::Preprocessed(provider) => provider.get_samples(data_ids).await,
        }
    }
}
