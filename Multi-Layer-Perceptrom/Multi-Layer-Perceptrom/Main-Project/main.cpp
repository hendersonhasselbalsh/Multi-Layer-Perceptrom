#include <opencv2/opencv.hpp>
#include "gnuplot-include.h"
#include "../utils/basic-includes.h"
#include "../mlp/multy-layer-perceptron.h"
 

std::vector<TrainigData> LoadData(const std::string& folderPath)
{
    //std::vector<MLP_DATA> set;
    std::vector<TrainigData> set;

    int l = -1;

    for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
        if (std::filesystem::is_regular_file(entry.path())) {

            std::string fileName = entry.path().filename().string();
            std::string labelStr = Utils::SplitString(fileName, "_")[0];
            //size_t label = (size_t)std::stoi(labelStr);    // MLP

            size_t labelIndex = (size_t)std::stoi(labelStr);                    // LSTM
            std::vector<double> label = std::vector<double>((size_t)10, 0.0);   // LSTM
            label[labelIndex]  =  1.0;                                          // LSTM

            std::string fullPathName = entry.path().string();
            Eigen::MatrixXd imgMat = Utils::ImageToMatrix(cv::imread(fullPathName));

            std::vector<double> input  =  Utils::FlatMatrix(imgMat);

            set.push_back({ input, label });

            if (labelIndex != l) {
                l = labelIndex;
                std::cout << "load data: [" << (labelIndex+1)*10 << "%]\n";
            }
        }
    }

    return set;
};

Eigen::MatrixXd TestingModelAccuracy(MLP* lstm, std::vector<TrainigData> testSet, double* accuracy)  // "..\\..\\.resources\\test"
{
    Eigen::MatrixXd confusionMatrix = Eigen::MatrixXd::Zero(10, 10);
    int totalData = 0;
    int errors = 0;

    for (auto& testData : testSet) {

        std::vector<double> givenOutput = lstm->Foward(testData.INPUT);

        auto it = std::max_element(givenOutput.begin(), givenOutput.end());
        size_t givenLabel = std::distance(givenOutput.begin(), it);

        auto it2 = std::max_element(testData.LABEL.begin(), testData.LABEL.end());
        size_t labelIndex = std::distance(testData.LABEL.begin(), it2);

        confusionMatrix(givenLabel, labelIndex) += 1.0;

        totalData++;

        if (givenLabel != labelIndex) { errors++; }

    }

    (*accuracy) = 1.0 - ((double)errors/totalData);

    return confusionMatrix;
}



int main(int argc, const char** argv)
{
    //--- initialize gnuplot to plot chart
    Gnuplot gnuplot;
    gnuplot.OutFile("..\\..\\res.dat");
    gnuplot.xRange("0", "");
    gnuplot.yRange("-0.01", "1.05");
    gnuplot.Grid("1", "0.1");


    //--- load MNIST training set
    std::vector<TrainigData> trainigDataSet  =  LoadData("..\\..\\MNIST_dataset\\train");
    std::vector<TrainigData> testDataSet  =  LoadData("..\\..\\MNIST_dataset\\test");


    //--- build mlp architecture and hiperparam
    MLP mlp  =  MlpBuilder()
        .InputSize(28*28)
        .Architecture({
            LayerSignature(128, new LeakyReLU(), 0.001),
            LayerSignature(10, new Sigmoid(2.0), 0.001),
        })
        .MaxEpochs(30)
        .LostFunction(new MSE())
        .SaveOn("..\\..\\mlp-architecture.json")
        .Build();


//--- training model, and do a callback on each epoch
    int epoch = 0;

    mlp.Training(trainigDataSet, [&]() {

        double accuracyTraining = 0.0;
        double accuracyTest = 0.0;

        Eigen::MatrixXd trainConfusionMatrix  =  TestingModelAccuracy(&mlp, trainigDataSet, &accuracyTraining);
        Eigen::MatrixXd testConfusionMatrix  =  TestingModelAccuracy(&mlp, testDataSet, &accuracyTest);

        std::cout << "\n\n------------------------------- epoch: " << epoch << " ------------------------------- \n\n";
        std::cout << "Training Accuracy: " << accuracyTraining << "\n\n";
        std::cout << trainConfusionMatrix << "\n\n";

        std::cout << "Test Accuracy: " << accuracyTest << "\n\n";
        std::cout << testConfusionMatrix << "\n\n\n";

        gnuplot.out << epoch << " " << accuracyTraining << " " << accuracyTest <<"\n";

        epoch++;
    });


    //--- plot chart
    gnuplot.out.close();
    gnuplot << "plot \'..\\..\\res.dat\' using 1:2 w l title \"Training Accuracy\", ";
    gnuplot << "\'..\\..\\res.dat\' using 1:3 w l title \"Training Accuracy\" \n";
    gnuplot << " \n";


    std::cout << "\n\n[SUCESSO]";
    return 0;
}

