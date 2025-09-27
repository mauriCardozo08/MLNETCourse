using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;

namespace MlNETProjects.ClassificationAI
{
    public class ClassificationAIForSentimentalAnalysis
    {
        public ClassificationAIForSentimentalAnalysis() { }

        public void Execute()
        {
            //TrainTheModel();
            TestModel();
        }

        #region Testing the model

        private void TestModel() 
        {
            string basePath = "C:\\Users\\mcardozo\\source\\repos\\MlNETProjects\\MlNETProjects\\ClassificationAI";
            string modelPath = Path.Combine(basePath, "model.zip");
            string testDataPath = Path.Combine(basePath, "test.csv");

            MLContext mLContext = new MLContext();
            ITransformer model;
            using(var stream = new FileStream(modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                model = mLContext.Model.Load(stream, out var modelInputSchema);
            }

            IDataView testData = mLContext.Data.LoadFromTextFile<TextData>(testDataPath, hasHeader: true, separatorChar: ',');

            var predictor = mLContext.Model.CreatePredictionEngine<TextData, SentimentPrediction>(model);
            var testDataList = mLContext.Data.CreateEnumerable<TextData>(testData, reuseRowObject: false).ToList();

            foreach (var data in testDataList) 
            {
                var prediction = predictor.Predict(data);
                Console.WriteLine($"Text:{data.text} | Positive Sentiment{prediction.IsPositiveSentiment}");
            }
        }
    

        public class TextData 
        {
            [LoadColumn(0)]
            public string text { get; set; }
        }

        public class SentimentPrediction 
        {
            [ColumnName("Score")]
            public float SentimentScore { get; set; }

            public bool IsPositiveSentiment => SentimentScore < 0.5f;
        }

        #endregion


        #region Trainning the model
        private void TrainTheModel() 
        {
            MLContext mLContext = new MLContext();
            string dataPath = "C:\\Users\\mcardozo\\source\\repos\\MlNETProjects\\MlNETProjects\\ClassificationAI\\train.csv";
            string text = File.ReadAllText(dataPath);

            using (StreamReader sr = new StreamReader(dataPath))
            {
                text = text.Replace("\'", "");
            }
            File.WriteAllText(dataPath, text);

            IDataView dataView = mLContext.Data.LoadFromTextFile<MovieReview>(dataPath, hasHeader: true, allowQuoting: true, separatorChar: ',');

            var pipeline = mLContext.Transforms.Text.FeaturizeText("Features", "text")
                .Append(mLContext.BinaryClassification.Trainers.SdcaLogisticRegression("Label", "Features"));

            var model = pipeline.Fit(dataView);
            var predictions = model.Transform(dataView);
            var metrics = mLContext.BinaryClassification.Evaluate(predictions, "Label");
            Console.WriteLine($"Accuracy: {metrics.Accuracy}");
            Console.WriteLine($"Precision: {metrics.PositivePrecision}");
            Console.WriteLine($"Recall: {metrics.PositiveRecall}");
            Console.WriteLine($"F1Score: {metrics.F1Score}");
            mLContext.Model.Save(model, dataView.Schema, "C:\\Users\\mcardozo\\source\\repos\\MlNETProjects\\MlNETProjects\\ClassificationAI\\model.zip");
        }

        public class MovieReview 
        {
            [LoadColumn(0)]
            public string text { get; set; }

            [LoadColumn(1)]
            [ColumnName("Label")]
            public bool sentiment { get; set; }
        }

        #endregion
    }
}
