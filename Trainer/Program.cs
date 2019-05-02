using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using System.Collections.Generic;

namespace Trainer
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("TACO TIME!");
            var baseFolder = Path.GetFullPath(@"..\..\..\..\data");

            var trainFolder = Path.Combine(baseFolder, "train");
            var valFolder = Path.Combine(baseFolder, "val");
            var modelLocation = Path.Combine(baseFolder, "tacomodel.zip");

            Console.WriteLine($"Using \"{trainFolder}\" for training images.");
            Console.WriteLine($"Using \"{valFolder}\" for validation images.");
            

            var start = DateTime.Now;
            // TRAINING TIME
            Train(trainFolder, modelLocation);
            Console.WriteLine($"Took to train {(DateTime.Now - start).TotalSeconds.ToString()}s");
            Console.WriteLine($"Saved model to \"{modelLocation}\"");


            Validation(valFolder, modelLocation);
        }

        static void Train(string trainingFolder, string modelLocation)
        {
            var context = new MLContext();
            var data = context.Data.LoadFromEnumerable(ImageData.ReadFromFolder(trainingFolder));
            var tfModel = context.Model.LoadTensorFlowModel("tensorflow_inception_graph.pb");

            var pipeline = context.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", inputColumnName: "Label")
                .Append(context.Transforms.LoadImages(outputColumnName: "input", 
                                                      imageFolder: "", 
                                                      inputColumnName: nameof(ImageData.Location)))
                .Append(context.Transforms.ResizeImages(outputColumnName: "input", 
                                                        imageWidth: ImageNetSettings.imageWidth, 
                                                        imageHeight: ImageNetSettings.imageHeight, 
                                                        inputColumnName: "input"))
                .Append(context.Transforms.ExtractPixels(outputColumnName: "input", 
                                                         interleavePixelColors: ImageNetSettings.channelsLast, 
                                                         offsetImage: ImageNetSettings.mean))
                .Append(tfModel.ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2_pre_activation" }, 
                                                     inputColumnNames: new[] { "input" },
                                                     addBatchDimensionInput: true))
                .Append(context.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "Label", 
                                                                                      featureColumnName: "softmax2_pre_activation"))
                .Append(context.Transforms.Conversion.MapKeyToValue("Prediction", "PredictedLabel"));

            ITransformer model = pipeline.Fit(data);
            context.Model.Save(model, data.Schema, modelLocation);

        }

        static void Validation(string valFolder, string modelLocation)
        {
            var context = new MLContext();
            var model = context.Model.Load(modelLocation, out var schema);
            var data = context.Data.LoadFromEnumerable(ImageData.ReadFromFolder(valFolder));

            // do the thing!
            var valData = model.Transform(data);

            // print out results
            var predictions = context.Data.CreateEnumerable<ImageDataPrediction>(valData, false, true);
            foreach (var pr in predictions)
                Console.WriteLine($"{pr.Location}, {pr.Prediction}, {pr.Score.Max()}");

            // print metrics
            var classificationContext = context.MulticlassClassification;
            var metrics = classificationContext.Evaluate(valData, labelColumnName: "Label", predictedLabelColumnName: "PredictedLabel");
            Console.WriteLine($"LogLoss is: {metrics.LogLoss}");
            Console.WriteLine($"PerClassLogLoss is: {String.Join(" , ", metrics.PerClassLogLoss.Select(c => c.ToString()))}");
        }
    }

    public struct ImageNetSettings
    {
        public const int imageHeight = 224;
        public const int imageWidth = 224;
        public const float mean = 117;
        public const float scale = 1;
        public const bool channelsLast = true;
    }

    public class ImageData
    {
        public string Location { get; set; }
        public string Label { get; set; }

        public static IEnumerable<ImageData> ReadFromFolder(string folder)
        {
            foreach (var name in Directory.EnumerateDirectories(folder))
            {
                var label = Path.GetFileName(name);
                foreach (var f in Directory.EnumerateFiles(name, "*.jpg"))
                    yield return new ImageData { Location = Path.GetFullPath(f), Label = label };
            }
        }
    }

    public class ImageDataPrediction
    {
        public string Location { get; set; }
        public float[] Score { get; set; }
        public string Prediction { get; set; }
    }
}
