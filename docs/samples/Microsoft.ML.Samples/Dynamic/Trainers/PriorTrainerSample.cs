﻿using System;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace Microsoft.ML.Samples.Dynamic
{
    public class PriorTrainerSample
    {
        public static void Example()
        {
            // Downloading the dataset from github.com/dotnet/machinelearning.
            // This will create a sentiment.tsv file in the filesystem.
            // You can open this file, if you want to see the data. 
            string dataFile = SamplesUtils.DatasetUtils.DownloadSentimentDataset();

            // A preview of the data. 
            // Sentiment	SentimentText
            //      0	    " :Erm, thank you. "
            //      1	    ==You're cool==

            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Step 1: Read the data as an IDataView.
            // First, we define the reader: specify the data columns and where to find them in the text file.
            var reader = mlContext.Data.CreateTextLoader(
                columns: new[]
                    {
                        new TextLoader.Column("Sentiment", DataKind.R4, 0),
                        new TextLoader.Column("SentimentText", DataKind.Text, 1)
                    },
                hasHeader: true
            );
            
            // Read the data
            var data = reader.Read(dataFile);

            // Split it between training and test data
            var (train, test) = mlContext.BinaryClassification.TrainTestSplit(data);

            // ML.NET doesn't cache data set by default. Therefore, if one reads a data set from a file and accesses it many times, 
            // it can be slow due to expensive featurization and disk operations. When the considered data can fit into memory, a 
            // solution is to cache the data in memory. Caching is especially helpful when working with iterative algorithms which 
            // needs many data passes. Since SDCA is the case, we cache. Inserting a cache step in a pipeline is also possible,
            // please see the construction of pipeline below.
            data = mlContext.Data.Cache(data);

            // Step 2: Pipeline 
            // Featurize the text column through the FeaturizeText API. 
            // Then append a binary classifier, setting the "Label" column as the label of the dataset, and 
            // the "Features" column produced by FeaturizeText as the features column.
            var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", "SentimentText")
                    .AppendCacheCheckpoint(mlContext) // Add a data-cache step within a pipeline.
                    .Append(mlContext.BinaryClassification.Trainers.Prior(labelColumn: "Sentiment"));

            // Step 3: Train the pipeline
            var trainedPipeline = pipeline.Fit(train);

            // Step 4: Evaluate on the test set
            var transformedData = trainedPipeline.Transform(test);
            var evalMetrics = mlContext.BinaryClassification.Evaluate(transformedData, label: "Sentiment");

            // Step 5: Inspect the output
            Console.WriteLine("Accuracy: " + evalMetrics.Accuracy);

            // Expected output:
            // Accuracy: 0.647058823529412
        }
    }
}
