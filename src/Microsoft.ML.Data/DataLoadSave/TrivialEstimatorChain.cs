// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    public sealed class TrivialEstimatorChain<TLastTransformer> : IEstimator<TransformerChain<TLastTransformer>>, ITransformer
        where TLastTransformer : class, ITransformer
    {
        private readonly IHost _host;
        private readonly EstimatorChain<TLastTransformer> _estimatorChain;
        private readonly TransformerChain<TLastTransformer> _transformerChain;

        private TrivialEstimatorChain(IHostEnvironment env, EstimatorChain<TLastTransformer> estimatorChain, TransformerChain<TLastTransformer> transformerChain)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(TrivialEstimatorChain<TLastTransformer>));

            _host.CheckValue(estimatorChain, nameof(estimatorChain));
            _host.CheckValue(transformerChain, nameof(transformerChain));

            _estimatorChain = estimatorChain;
            _transformerChain = transformerChain;
        }

        public TrivialEstimatorChain()
        {

        }

        public TrivialEstimatorChain<ITransformer> Append<TTrivialEstimator>(TTrivialEstimator estimator, TransformerScope scope = TransformerScope.Everything)
            where TTrivialEstimator : class, IEstimator<ITransformer>, ITransformer
            => new TrivialEstimatorChain<ITransformer>(_host, _estimatorChain.Append(estimator, scope), _transformerChain.Append(estimator as ITransformer));

        public EstimatorChain<TNewTrans> Append<TNewTrans>(IEstimator<TNewTrans> estimator, TransformerScope scope = TransformerScope.Everything)
            where TNewTrans : class, ITransformer
            => _estimatorChain.Append(estimator, scope);
        // REVIEW: Should we also allow ITransformer to be appended to the TrivialEstimatorChain thereby producing a TransformerChain?

        /// <summary>
        /// Append a 'caching checkpoint' to the estimator chain. This will ensure that the downstream estimators will be trained against
        /// cached data. It is helpful to have a caching checkpoint before trainers that take multiple data passes.
        /// </summary>
        /// <param name="env">The host environment to use for caching.</param>
        public TrivialEstimatorChain<TLastTransformer> AppendCacheCheckpoint(IHostEnvironment env)
            => new TrivialEstimatorChain<TLastTransformer>(env, _estimatorChain.AppendCacheCheckpoint(env), _transformerChain);

        public TransformerChain<TLastTransformer> Fit(IDataView input)
        {
            _host.CheckValue(input, nameof(input));
            return _transformerChain;
        }

        public IDataView Transform(IDataView input)
        {
            _host.CheckValue(input, nameof(input));
            return _transformerChain.Transform(input);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
            => _estimatorChain.GetOutputSchema(inputSchema);

        public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
            => _transformerChain.GetOutputSchema(inputSchema);

        bool ITransformer.IsRowToRowMapper => ((ITransformer)_transformerChain).IsRowToRowMapper;

        IRowToRowMapper ITransformer.GetRowToRowMapper(DataViewSchema inputSchema)
           => ((ITransformer)_transformerChain).GetRowToRowMapper(inputSchema);

        void ICanSaveModel.Save(ModelSaveContext ctx)
            => ((ITransformer)_transformerChain).Save(ctx);
    }
}
