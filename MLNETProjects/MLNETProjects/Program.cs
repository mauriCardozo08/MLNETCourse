using MlNETProjects.ClassificationAI;
using MlNETProjects.RegressionAI;
using MLNETProjects.ForecastingAI;
using MLNETProjects.ImageClassificator;
using MLNETProjects.RecommendationAI;

//Console.WriteLine("Starting Image classificator AI");
//ImageClassificator imgClassificator = new ImageClassificator();
//imgClassificator.TrainAndTestModel();

//Console.WriteLine("Starting Regression AI");
//RegressionAI regressionAI = new RegressionAI();
//regressionAI.Execute();

//Console.WriteLine("Starting Classification AI");
//ClassificationAIForSentimentalAnalysis classificationAIForSentimentalAnalysis = new ClassificationAIForSentimentalAnalysis();
//classificationAIForSentimentalAnalysis.Execute();

//Console.WriteLine("Starting Forecasting AI");
//ForecastingAI forecastingAI = new ForecastingAI();
//forecastingAI.Execute();

Console.WriteLine("Starting Recommendation AI");
RecommendationAI recommendationAI = new RecommendationAI();
recommendationAI.Execute();

Console.WriteLine();
