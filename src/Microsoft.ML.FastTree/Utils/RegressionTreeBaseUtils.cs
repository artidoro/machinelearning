﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.FastTree;

namespace Microsoft.ML.FastTree.Utils
{
    internal class RegressionTreeBaseUtils
    {
        /// <summary>
        /// Utility method used to represent a tree ensemble as an <see cref="IDataView"/>.
        /// Every row in the <see cref="IDataView"/> corresponds to a node in the tree ensemble. The columns are the fields for each node.
        /// The column TreeID specifies which tree the node belongs to. The <see cref="QuantileRegressionTree"/> gets
        /// special treatment since it has some additional fields (<see cref="QuantileRegressionTree.GetLeafSamplesAt(int)"/>
        /// and <see cref="QuantileRegressionTree.GetLeafSampleWeightsAt(int)"/>).
        /// </summary>
        public static IDataView RegressionTreeEnsembleAsIDataView(IHost host, double bias, IReadOnlyList<double> treeWeights, IReadOnlyList<RegressionTreeBase> trees)
        {
            var builder = new ArrayDataViewBuilder(host);
            var numberOfRows = trees.Select(tree => tree.NumberOfNodes).Sum() + trees.Select(tree => tree.NumberOfLeaves).Sum();

            // Bias column. This will be a repeated value for all rows in the resulting IDataView.
            builder.AddColumn("Bias", NumberDataViewType.Double, Enumerable.Repeat(bias, numberOfRows).ToArray());

            // TreeWeights column. The TreeWeight value will be repeated for all the notes in the same tree in the IDataView.
            var treeWeightsList = new List<double>();
            for (int i = 0; i < trees.Count; i++)
                treeWeightsList.AddRange(Enumerable.Repeat(treeWeights[i], trees[i].NumberOfNodes + trees[i].NumberOfLeaves));

            builder.AddColumn("TreeWeights", NumberDataViewType.Double, treeWeightsList.ToArray());

            // Tree id indicates which tree the node belongs to.
            var treeId = new List<int>();
            for (int i = 0; i < trees.Count; i++)
                treeId.AddRange(Enumerable.Repeat(i, trees[i].NumberOfNodes + trees[i].NumberOfLeaves));

            builder.AddColumn("TreeID", NumberDataViewType.Int32, treeId.ToArray());

            // IsLeaf column indicates if node is a leaf node.
            var isLeaf = new List<ReadOnlyMemory<char>>();
            for (int i = 0; i < trees.Count; i++)
            {
                isLeaf.AddRange(Enumerable.Repeat(new ReadOnlyMemory<char>("Tree node".ToCharArray()), trees[i].NumberOfNodes));
                isLeaf.AddRange(Enumerable.Repeat(new ReadOnlyMemory<char>("Leaf node".ToCharArray()), trees[i].NumberOfLeaves));
            }

            // Bias column. This is a repeated value for all trees.
            builder.AddColumn("IsLeaf", TextDataViewType.Instance, isLeaf.ToArray());

            // LeftChild column.
            var leftChild = new List<int>();
            for (int i = 0; i < trees.Count; i++)
            {
                leftChild.AddRange(trees[i].LeftChild.AsEnumerable());
                leftChild.AddRange(Enumerable.Repeat(0, trees[i].NumberOfLeaves));
            }

            builder.AddColumn(nameof(RegressionTreeBase.LeftChild), NumberDataViewType.Int32, leftChild.ToArray());

            // RightChild column.
            var rightChild = new List<int>();
            for (int i = 0; i < trees.Count; i++)
            {
                rightChild.AddRange(trees[i].RightChild.AsEnumerable());
                rightChild.AddRange(Enumerable.Repeat(0, trees[i].NumberOfLeaves));
            }

            builder.AddColumn(nameof(RegressionTreeBase.RightChild), NumberDataViewType.Int32, rightChild.ToArray());

            // NumericalSplitFeatureIndexes column.
            var numericalSplitFeatureIndexes = new List<int>();
            for (int i = 0; i < trees.Count; i++)
            {
                numericalSplitFeatureIndexes.AddRange(trees[i].NumericalSplitFeatureIndexes.AsEnumerable());
                numericalSplitFeatureIndexes.AddRange(Enumerable.Repeat(0, trees[i].NumberOfLeaves));
            }

            builder.AddColumn(nameof(RegressionTreeBase.NumericalSplitFeatureIndexes), NumberDataViewType.Int32, numericalSplitFeatureIndexes.ToArray());

            // NumericalSplitThresholds column.
            var numericalSplitThresholds = new List<float>();
            for (int i = 0; i < trees.Count; i++)
            {
                numericalSplitThresholds.AddRange(trees[i].NumericalSplitThresholds.AsEnumerable());
                numericalSplitThresholds.AddRange(Enumerable.Repeat(0f, trees[i].NumberOfLeaves));
            }

            builder.AddColumn(nameof(RegressionTreeBase.NumericalSplitThresholds), NumberDataViewType.Single, numericalSplitThresholds.ToArray());

            // CategoricalSplitFlags column.
            var categoricalSplitFlags = new List<bool>();
            for (int i = 0; i < trees.Count; i++)
            {
                categoricalSplitFlags.AddRange(trees[i].CategoricalSplitFlags.AsEnumerable());
                categoricalSplitFlags.AddRange(Enumerable.Repeat(false, trees[i].NumberOfLeaves));
            }

            builder.AddColumn(nameof(RegressionTreeBase.CategoricalSplitFlags), BooleanDataViewType.Instance, categoricalSplitFlags.ToArray());

            // LeafValues column.
            var leafValues = new List<double>();
            for (int i = 0; i < trees.Count; i++)
            {
                leafValues.AddRange(Enumerable.Repeat(0d, trees[i].NumberOfNodes));
                leafValues.AddRange(trees[i].LeafValues.AsEnumerable());
            }

            builder.AddColumn(nameof(RegressionTreeBase.LeafValues), NumberDataViewType.Double, leafValues.ToArray());

            // SplitGains column.
            var splitGains = new List<double>();
            for (int i = 0; i < trees.Count; i++)
            {
                splitGains.AddRange(trees[i].SplitGains.AsEnumerable());
                splitGains.AddRange(Enumerable.Repeat(0d, trees[i].NumberOfLeaves));
            }

            builder.AddColumn(nameof(RegressionTreeBase.SplitGains), NumberDataViewType.Double, splitGains.ToArray());

            // CategoricalSplitFeatures column.
            var categoricalSplitFeatures = new List<VBuffer<int>>();
            for (int i = 0; i < trees.Count; i++)
            {
                for (int j = 0; j < trees[i].NumberOfNodes; j++)
                {
                    var categoricalSplitFeaturesArray = trees[i].GetCategoricalSplitFeaturesAt(j).ToArray();
                    categoricalSplitFeatures.Add(new VBuffer<int>(categoricalSplitFeaturesArray.Length, categoricalSplitFeaturesArray));
                    var len = trees[i].GetCategoricalSplitFeaturesAt(j).ToArray().Length;
                }

                categoricalSplitFeatures.AddRange(Enumerable.Repeat(new VBuffer<int>(), trees[i].NumberOfLeaves));
            }

            builder.AddColumn("CategoricalSplitFeatures", NumberDataViewType.Int32, categoricalSplitFeatures.ToArray());

            // CategoricalCategoricalSplitFeatureRange column.
            var categoricalCategoricalSplitFeatureRange = new List<VBuffer<int>>();
            for (int i = 0; i < trees.Count; i++)
            {
                for (int j = 0; j < trees[i].NumberOfNodes; j++)
                {
                    var categoricalCategoricalSplitFeatureRangeArray = trees[i].GetCategoricalCategoricalSplitFeatureRangeAt(j).ToArray();
                    categoricalCategoricalSplitFeatureRange.Add(new VBuffer<int>(categoricalCategoricalSplitFeatureRangeArray.Length, categoricalCategoricalSplitFeatureRangeArray));
                    var len = trees[i].GetCategoricalCategoricalSplitFeatureRangeAt(j).ToArray().Length;
                }
                categoricalCategoricalSplitFeatureRange.AddRange(Enumerable.Repeat(new VBuffer<int>(), trees[i].NumberOfLeaves));
            }

            builder.AddColumn("CategoricalCategoricalSplitFeatureRange", NumberDataViewType.Int32, categoricalCategoricalSplitFeatureRange.ToArray());

            // If the input tree array is a quantile regression tree we need to add two more columns.
            var quantileTrees = trees as IReadOnlyList<QuantileRegressionTree>;
            if (quantileTrees != null)
            {
                // LeafSamples column.
                var leafSamples = new List<VBuffer<double>>();
                for (int i = 0; i < quantileTrees.Count; i++)
                {
                    leafSamples.AddRange(Enumerable.Repeat(new VBuffer<double>(), quantileTrees[i].NumberOfNodes));
                    for (int j = 0; j < quantileTrees[i].NumberOfLeaves; j++)
                    {
                        var leafSamplesArray = quantileTrees[i].GetLeafSamplesAt(j).ToArray();
                        leafSamples.Add(new VBuffer<double>(leafSamplesArray.Length, leafSamplesArray));
                        var len = quantileTrees[i].GetLeafSamplesAt(j).ToArray().Length;
                    }
                }

                builder.AddColumn("LeafSamples", NumberDataViewType.Double, leafSamples.ToArray());

                // LeafSampleWeights column.
                var leafSampleWeights = new List<VBuffer<double>>();
                for (int i = 0; i < quantileTrees.Count; i++)
                {
                    leafSampleWeights.AddRange(Enumerable.Repeat(new VBuffer<double>(), quantileTrees[i].NumberOfNodes));
                    for (int j = 0; j < quantileTrees[i].NumberOfLeaves; j++)
                    {
                        var leafSampleWeightsArray = quantileTrees[i].GetLeafSampleWeightsAt(j).ToArray();
                        leafSampleWeights.Add(new VBuffer<double>(leafSampleWeightsArray.Length, leafSampleWeightsArray));
                        var len = quantileTrees[i].GetLeafSampleWeightsAt(j).ToArray().Length;
                    }
                }

                builder.AddColumn("LeafSampleWeights", NumberDataViewType.Double, leafSampleWeights.ToArray());
            }

            var data = builder.GetDataView();
            return data;
        }

    }
}
