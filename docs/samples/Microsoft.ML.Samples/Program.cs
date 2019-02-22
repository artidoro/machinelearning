using System;
using Microsoft.ML.Samples.Dynamic;
using Microsoft.ML.Trainers;

namespace Microsoft.ML.Samples
{
    class Program
    {
        static void Main(string[] args)
        {
            var ml = new MLContext();

            // Set up logging to print to console.
            ml.Log += (sender, e) => Console.WriteLine(e.Message);

            // Read data.
            var data = ml.Data.ReadFromBinary("../../../../../mlnettorch/traintestvalid/Criteo_test_1M.bin");

            //var data = ml.Data.TakeRows(ml.Data.ReadFromBinary("../../../../../mlnettorch/traintestvalid/Criteo_test_1M.bin"), 1000);
            //var data = ml.Data.ReadFromBinary("../../../data/Criteo_train_43M.bin");
            // If data is cached no call to rowshuffling transformer.
            //var cachedData = ml.Data.Cache(data);

            // Define pipeline.
            var pipeline = ml.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(new SdcaMultiClassTrainer.Options
            {
                NumThreads = 1,
                MaxIterations = 1000
            })
            .AppendCacheCheckpoint(ml);


            // Loop over training 1000 times to trigger deadlock.
            for (int i = 0; i < 1000; i++)
            {
                var transformer = pipeline.Fit(data);

                var transformedData = transformer.Transform(data);

                var metrics = ml.MulticlassClassification.Evaluate(transformedData);

                Console.WriteLine("ACC:" + metrics.AccuracyMacro.ToString());
                Console.WriteLine($"Done {i}");
            }

            Console.ReadLine();
        }
    }
}
