using Microsoft.ML;
using Microsoft.ML.Data;
using System.Diagnostics;

//чекпоинт до вмешательства 1
//Если файл не работает, ставим абсолютный путь от диска до папки назначения | 
string trainFilePath = "Data.tsv";
string modelFilepath = "model.zip";

IDataView trainingDataV;
MLContext mlContext;
ITransformer model;
mlContext = new MLContext(seed:0); //seed:0 это что-то на подобии сидов генерации, но это не точно 

//инициализациия загрузки данных из файла, где мы передаём значение пути файла и булево значения на то, имеет ли файл заголовки
trainingDataV = mlContext.Data.LoadFromTextFile<WeatherReport>(trainFilePath,hasHeader:true );

var pipline = ProcessData();

var training_pipline = BuildTrainModel(trainingDataV,pipline);

SaveModel();

var keeprun = true;
Console.WriteLine("Введите данные о погоде в стиле /Сегодня холодно/. Чтобы закончить введите End");

while (keeprun)
{
    var subjectline = Console.ReadLine();

    if(subjectline == "End")
    {
        keeprun = false;
    }
    else
    {
        Console.WriteLine(PredictRecommendation(subjectline));    
    }
}

Console.ReadLine();
void SaveModel()
{
    mlContext.Model.Save(model, trainingDataV.Schema,modelFilepath);
}

PredictionEngine<WeatherReport, Recomendation> predictionEngine;

string PredictRecommendation(string subjectline)
{
    var model = mlContext.Model.Load(modelFilepath, out var modeinputscheme);
    var weatherSubject = new WeatherReport() { Weather = subjectline };
    predictionEngine = mlContext.Model.CreatePredictionEngine<WeatherReport, Recomendation>(model);
    var result = predictionEngine.Predict(weatherSubject);
    return result.recommendation;
}

IEstimator<ITransformer> ProcessData()
{
    var pipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Погода", outputColumnName: "Рекомендация")
        .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "СубъектПогода", outputColumnName: "СубъектРекомендации")
        .Append(mlContext.Transforms.Concatenate("Функция", "ФункцияРекПогода"))
       .AppendCacheCheckpoint(mlContext));

    return pipeline;
}

IEstimator<ITransformer> BuildTrainModel(IDataView trainingDataV, IEstimator<ITransformer> pipline)
{
    //тут очень интересный момент, дело в том, что нужно выбрать то, как мы будем классифицировать данные,
    //то есть, у нас есть вариант бниарной классификации, где объект может иметь лишь одно конкретное определение и больше ничего, либо, объект
    // может иметь мультиклассовую классификацию
    var trainingPipe = pipline.Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Функция"))
        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
    model = trainingPipe.Fit(trainingDataV);
    return trainingPipe;
}
public class WeatherReport()
{
    [LoadColumn(0)]
    public string  Weather { get; set; }
    [LoadColumn(1)]
    public string  Recomendation { get; set; }
}

public class Recomendation
{
    [ColumnName("PredictedLabel")]
    public string recommendation { get; set; }
}