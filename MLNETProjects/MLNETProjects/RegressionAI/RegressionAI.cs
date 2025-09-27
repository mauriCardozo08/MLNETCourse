using Microsoft.ML;
using Microsoft.ML.Data;

namespace MlNETProjects.RegressionAI
{
    public class RegressionAI
    {
        public RegressionAI() 
        {

        }

        public void Execute()
        {
            TrainAndTestTheModel();
        }

        private void TrainAndTestTheModel()
        {
            var mlContext = new MLContext(seed: 0);
            var dataPath = "C:\\Users\\mcardozo\\source\\repos\\MlNETProjects\\MlNETProjects\\RegressionAI\\house-price-data.csv";

            IDataView dataView = mlContext.Data.LoadFromTextFile<HouseData>(dataPath, separatorChar: ',', hasHeader: true);
            var trainTestData = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            var trainData = trainTestData.TrainSet;
            var testData = trainTestData.TestSet;

            var pipeline = mlContext.Transforms.Concatenate("Features", "HouseSize", "NumberOfBedrooms", "NumberOfBathrooms")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Neighborhood"))
                .Append(mlContext.Transforms.Concatenate("Features", "Features", "Neighborhood")
                .Append(mlContext.Transforms.CopyColumns("Label", "Price"))
                .Append(mlContext.Regression.Trainers.FastTree(labelColumnName: "Label")));

            var trainedModel = pipeline.Fit(trainData);

            var predictions = trainedModel.Transform(testData);
            var metrics = mlContext.Regression.Evaluate(predictions);

            Console.WriteLine($"R^2: {metrics.RSquared:0.##}");
            Console.WriteLine($"RMS: {metrics.RootMeanSquaredError:#.##}");

            var predictionEngine = mlContext.Model.CreatePredictionEngine<HouseData, HousePricePrediction>(trainedModel);

            var sampleHouse = new HouseData()
            {
                HouseSize = 2500f,
                NumberOfBedrooms = 4f,
                NumberOfBathrooms = 3f,
                Neighborhood = "Northeast"
            };
            var pricePrediction = predictionEngine.Predict(sampleHouse);
            Console.WriteLine($"Predicted price for the house: {pricePrediction.PredictedSalePrice:C}");
        }
    }

    public class HouseData 
    {
        [LoadColumn(0)]
        public float HouseSize { get; set; }

        [LoadColumn(1)]
        public float NumberOfBedrooms { get; set; }

        [LoadColumn(2)]
        public float NumberOfBathrooms { get; set; }

        [LoadColumn(3)]
        public string Neighborhood { get; set; }

        [LoadColumn(4)]
        public float Price { get; set; }
    }

    public class HousePricePrediction 
    {
        [ColumnName("Score")]
        public float PredictedSalePrice { get; set; }
    }
}
