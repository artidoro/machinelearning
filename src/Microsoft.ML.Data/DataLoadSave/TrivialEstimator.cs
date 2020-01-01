﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// The trivial implementation of <see cref="IEstimator{TTransformer}"/> that already has
    /// the transformer and returns it on every call to <see cref="Fit(IDataView)"/>.
    ///
    /// Concrete implementations still have to provide the schema propagation mechanism, since
    /// there is no easy way to infer it from the transformer.
    /// </summary>
    public abstract class TrivialEstimator<TTransformer> : IEstimator<TTransformer>, ITransformer
        where TTransformer : class, ITransformer
    {
        [BestFriend]
        private protected readonly IHost Host;
        [BestFriend]
        internal readonly TTransformer Transformer;

        bool ITransformer.IsRowToRowMapper => Transformer.IsRowToRowMapper;

        [BestFriend]
        private protected TrivialEstimator(IHost host, TTransformer transformer)
        {
            Contracts.AssertValue(host);
            Host = host;
            Host.CheckValue(transformer, nameof(transformer));
            Transformer = transformer;
        }

        public TTransformer Fit(IDataView input)
        {
            Host.CheckValue(input, nameof(input));
            // Validate input schema.
            Transformer.GetOutputSchema(input.Schema);
            return Transformer;
        }

        public IDataView Transform(IDataView input)
            => Transformer.Transform(input);

        public abstract SchemaShape GetOutputSchema(SchemaShape inputSchema);

        DataViewSchema ITransformer.GetOutputSchema(DataViewSchema inputSchema)
            => Transformer.GetOutputSchema(inputSchema);

        IRowToRowMapper ITransformer.GetRowToRowMapper(DataViewSchema inputSchema)
            => Transformer.GetRowToRowMapper(inputSchema);

        void ICanSaveModel.Save(ModelSaveContext ctx)
            => Transformer.Save(ctx);
    }
}
