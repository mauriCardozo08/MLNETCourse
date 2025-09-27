using System;
using System.CodeDom;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;
using static Microsoft.ML.DataOperationsCatalog;
using Tensorflow;


namespace MLNETProjects.ImageClassificator
{
    public class ImageClassificator
    {
        public ImageClassificator()
        {

        }

        public void Execute() {
            TrainAndTestModel();
        }

        public void TrainAndTestModel() {
            MLContext mlContext = new MLContext();

            IEnumerable<ImageData> images = LoadImagesFromDirectory(folder: dataFolder);
            IDataView imageData = mlContext.Data.LoadFromEnumerable(images);

            IDataView shuffledData = mlContext.Data.ShuffleRows(imageData);

            var preprocesingPipeline = mlContext.Transforms.Conversion
                .MapValueToKey(inputColumnName: "Label", outputColumnName: "LabelKey")
                .Append(mlContext.Transforms.LoadRawImageBytes("Image", dataFolder, "ImagePath"));

            IDataView preprocessedData = preprocesingPipeline.Fit(shuffledData).Transform(shuffledData);

            TrainTestData trainTestSplit = mlContext.Data.TrainTestSplit(preprocessedData, testFraction: 0.4);
            IDataView trainSet = trainTestSplit.TrainSet;
            IDataView testSet = trainTestSplit.TestSet;

            var classifierOptions = new ImageClassificationTrainer.Options()
            {
                FeatureColumnName = "Image",
                LabelColumnName = "LabelKey",
                Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
                MetricsCallback = Console.WriteLine,
                TestOnTrainSet = false,
                ValidationSet = testSet,
                ReuseTrainSetBottleneckCachedValues = true,
                ReuseValidationSetBottleneckCachedValues = true
            };

            var trainingPipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(classifierOptions)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            ITransformer trainedModel = trainingPipeline.Fit(trainSet);

            ClassifyMultiple(mlContext, testSet, trainedModel);
        }
        
        string dataFolder = Path.Combine(
              AppDomain.CurrentDomain.BaseDirectory,
                @"..\..\..\ImageClassificator\Data"
        );

        private static IEnumerable<ImageData> LoadImagesFromDirectory(string folder) 
        {
            var files = Directory.GetFiles(folder, "*", searchOption: SearchOption.AllDirectories);
            foreach (var file in files) 
            {
                if ((Path.GetExtension(file) != ".jpg") &&
                   (Path.GetExtension(file) != ".png") &&
                   (Path.GetExtension(file) != ".jpeg")) {
                    continue;
                }

                string label = Path.GetFileNameWithoutExtension(file).Trim();
                label = label.Substring(0, label.Length - 1);

                yield return new ImageData()
                {
                    ImagePath = file,
                    Label = label,
                };
            }
        }
        private static void PrintDataView(IDataView dataView) 
        {
            var preview = dataView.Preview();
            foreach(var row in preview.RowView) 
            {
                foreach (var kvp in row.Values) 
                {
                    Console.WriteLine($"{kvp.Key}: {kvp.Value} ");
                }
                Console.WriteLine();
            }
        }

        private static void OutputPrediction(Output prediction) 
        {
            string imageName = Path.GetFileName(prediction.ImagePath);
            Console.WriteLine($"Image: {imageName} | Actual Label: {prediction.Label} | Predicted Label: {prediction.PredictedLabel}");
        }

        private static void ClassifyMultiple(MLContext mlContext, IDataView data, ITransformer trainedModel) 
        {
            IDataView predictionData = trainedModel.Transform(data);
            var predictions = mlContext.Data.CreateEnumerable<Output>(predictionData, reuseRowObject:false).ToList();

            Console.WriteLine("AI predictions:");
            foreach (var prediction in predictions)
            {
                OutputPrediction(prediction);
            }
        }
    }

    public class ImageData
    {
        [LoadColumn(0)]
        public string? ImagePath { get; set; }

        [LoadColumn(1)]
        public string? Label { get; set; }
    }

    public class InputData
    {
        public byte[] Image { get; set; }
        public uint LabelKey { get; set; }
        public string ImagePath { get; set; }
        public string Label { get; set; }
    }

    public class Output 
    {
        public string ImagePath { get; set; }
        public string Label { get; set; }
        public string PredictedLabel { get; set; }
    }
}
