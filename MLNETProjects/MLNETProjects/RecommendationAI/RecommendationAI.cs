using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace MLNETProjects.RecommendationAI
{
    public class RecommendationAI
    {
        public RecommendationAI() { }

        public void Execute()
        {
            Console.WriteLine("RecommendationAI Execute method called.");
            ExecuteAndPrint();
        }

        private void ExecuteAndPrint()
        {
            string dataPath = Path.Combine(
                AppDomain.CurrentDomain.BaseDirectory,
                @"..\..\..\RecommendationAI\data\ratings.csv"
            );

            MLContext mLContext = new MLContext();
            IDataView fullData = mLContext.Data.LoadFromTextFile<MovieRating>(dataPath, separatorChar: ',', hasHeader: true);
            IDataView preprocessedData = PreprocessData(mLContext, fullData);
            SaveData(mLContext, preprocessedData, "preprocessed_rating.csv");
            (IDataView trainData, IDataView testData) data = LoadData(mLContext);
            PrintDataPreview(data.trainData);
            ITransformer model = TrainModel(mLContext, data.trainData);
        }
        private void PrintDataPreview() 
        {
        
        }

        private ITransformer TrainModel(MLContext mLContext, IDataView trainingDataView){
            
            IEstimator<ITransformer> estimator = mLContext.Transforms.Conversion
                .MapValueToKey(outputColumnName: "outputUserId", inputColumnName: "userId")
                .Append(mLContext.Transforms.Conversion.MapValueToKey(outputColumnName: "outputMovieId", inputColumnName: "movieId"));
            
            var options = new MatrixFactorizationTrainer.Options{
                MatrixColumnIndexColumnName = "outputUserId",
                MatrixRowIndexColumnName = "outputMovieId",
                LabelColumnName = "Label",
                NumberOfIterations = 10,
                ApproximationRank = 100,
            };

            var trainerEstimator = estimator.Append(
                mLContext.Recommendation().Trainers.MatrixFactorization(options)
            );

            ITransformer model = trainerEstimator.Fit(trainingDataView);
            Console.WriteLine("Model succesfully trained");

            return model;
        }

        private (IDataView training, IDataView test) LoadData(MLContext mLContext) 
        {
            var dataPath = "preprocessed_rating.csv";
            IDataView fullData = mLContext.Data.LoadFromTextFile<MovieRating>(dataPath, separatorChar: ',', hasHeader: true);
            var trainTestData = mLContext.Data.TrainTestSplit(fullData, testFraction: 0.2);

            IDataView trainData = trainTestData.TrainSet;
            IDataView testData = trainTestData.TestSet;
            return (trainData, testData);
        }

        public static void PrintDataPreview(IDataView dataview) 
        { 
            var preview = dataview.Preview();
            foreach (var row in preview.RowView) 
            {
                foreach (var column in row.Values) 
                {
                    Console.WriteLine($"{column.Key}: {column.Value}");
                }
                Console.WriteLine();
            }
        }

        private IDataView PreprocessData(MLContext mLContext, IDataView dataView) 
        {
            return mLContext.Transforms.Conversion.MapValueToKey(outputColumnName: "userId", inputColumnName: "userId")
                .Append(mLContext.Transforms.Conversion.MapValueToKey(outputColumnName: "movieId", inputColumnName: "movieId"))
                .Fit(dataView).Transform(dataView);
        }

        private void SaveData(MLContext mLContext, IDataView dataView, string dataPath) 
        {
            using(var fileStream = new FileStream(dataPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mLContext.Data.SaveAsText(dataView, fileStream, separatorChar: ',', headerRow: true, schema: true);
            }
        }

        public void EvalueteModel(MLContext mLContext, IDataView testDataView, ITransformer model) 
        {
            var prediction = model.Transform(testDataView);
            var metrics = mLContext.Regression.Evaluate(prediction, labelColumnName: "Label", scoreColumnName: "Score");
            Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError.ToString()}");
            Console.WriteLine($"RSquared: "+ metrics.RSquared.ToString());
        }

        public void UseModelForSinglePrediction(MLContext mLContext, ITransformer model)
        {
            var predictionEngine = mLContext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(model);
            var testInput = new MovieRating { userId = 14, movieId = 433 };
            var movieRatingPRediction = predictionEngine.Predict(testInput);
            Console.WriteLine($"Predicted rating for movie{testInput.movieId} is {Math.Round(movieRatingPRediction.Score,1)}");
            string recommendation = Math.Round(movieRatingPRediction.Score, 1) > 3.5 ?
                $"Movie {testInput.movieId} is recommended for user {testInput.userId}" : 
                $"Movie {testInput.movieId} is not recommended for user {testInput.userId}";
            Console.WriteLine(recommendation);
        }
    }

    public class MovieRating 
    {
        [LoadColumn(0)]
        public float userId { get; set; }
        [LoadColumn(1)]
        public float movieId { get; set; }
        [LoadColumn(2)]
        public float Label { get; set; }
    }

    public class MovieRatingPrediction 
    {
        public float Label;
        public float Score;
    
    }
}
