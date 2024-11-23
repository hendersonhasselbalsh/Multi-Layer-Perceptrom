#include "neuron.h"


template const auto Neuron::Get<Neuron::Attribute::ACTIVATION_FUNC>() const;
template const auto Neuron::Get<Neuron::Attribute::BIAS>() const;
template const auto Neuron::Get<Neuron::Attribute::ERROR>() const;
template const auto Neuron::Get<Neuron::Attribute::GRADIENT_DL_DU>() const;
template const auto Neuron::Get<Neuron::Attribute::LOST_FUNC>() const;
template const auto Neuron::Get<Neuron::Attribute::OUTPUT>() const;
template const auto Neuron::Get<Neuron::Attribute::U>() const;
template const auto Neuron::Get<Neuron::Attribute::WEIGHTS>() const;

template void Neuron::Set<Neuron::Attribute::ACTIVATION_FUNC, IActivationFunction*>(IActivationFunction* value);
template void Neuron::Set<Neuron::Attribute::LOST_FUNC, ILostFunction*>(ILostFunction* value);
template void Neuron::Set<Neuron::Attribute::LEARNING_RATE, double>(double value);




Neuron::Neuron(size_t inputSize, IActivationFunction* actFun, double leraningRate, ILostFunction* lostFunc, size_t outputSize)
    : _inputSize(inputSize+1), _learningRate(leraningRate), activationFunction(actFun), _lostFunction(lostFunc)
{
    assert(outputSize != 0);

    _error = 0.0;
    _u = 0.0;
    _delta = 0.0;

    _weights  =  Eigen::MatrixXd::Ones(1, _inputSize);
    _lossGradient  =  Eigen::MatrixXd::Ones(1, _inputSize);

    // uniform initialization
    /*for (auto& weight : _weights) {
        weight  =  Utils::RandomNormalDistributionValue(-1.0, 1.0);
    }*/
    //XavierWeightInitialization( _inputSize, outputSize );
}



Neuron::~Neuron() { }






double Neuron::CalculateOutput(Eigen::MatrixXd inputs)
{
    assert(inputs.size() == _weights.size());
    _u  =  (inputs * _weights.transpose()).array()(0);
    double output  =  activationFunction->f(_u);

    return output;
}

void Neuron::CalculateLossGradient(double correctValue, double predictedValue, Eigen::MatrixXd input)
{
    double partialActivated = _lostFunction->df(predictedValue, correctValue);      // (dL/da) -> loss partial with respect to activation
    double partialWeightedSum = activationFunction->df(_u);                         // (da/dU) -> activation partial with respect to weighted sum

    _delta = partialActivated * partialWeightedSum;                                 // (dL/dU) -> loss partial with respect to weighted sum

    _lossGradient  =  (partialActivated * partialWeightedSum) * input;              // (dL/dw) = (dL/da) * (da/dU) * (dU/dw)
   

    _error  =  _lostFunction->f(predictedValue, correctValue);
}



void Neuron::CalculateLossGradient(double LossPartialWithRespectToActivation, Eigen::MatrixXd input)
{
    double partialWeightedSum = activationFunction->df(_u);                                 // (da/dU) -> activation partial with respect to weighted sum

    _delta = LossPartialWithRespectToActivation * partialWeightedSum;                       // (dL/dU) -> loss partial with respect to weighted sum

    _lossGradient  =  (LossPartialWithRespectToActivation * partialWeightedSum) * input;    // (dL/dw) = (dL/da) * (da/dU) * (dU/dw)
}

double Neuron::LossPartialWithRespectToInput(size_t inputIndex)
{
    return _delta * _weights(0, inputIndex);      // (dL/dx) = (dL/dU) * (dU/dx) 
}



void Neuron::UpdateWeights()
{
    _weights  =  _weights - _learningRate * _lossGradient;
}



void Neuron::XavierWeightInitialization(size_t inputSize, size_t outputSize)
{
    double n_i = (double)inputSize;
    double n_o = (double)outputSize;

    double range = std::sqrt(6.0) / std::sqrt(n_i + n_o);

    for (size_t i = 0; i < inputSize; i++) {
        _weights(0, i)  =  Utils::RandomUniformDistribution(-range, range);
    }
}

double& Neuron::operator[](int weightIndex)
{
    return (double&)_weights(0,weightIndex);
}



//Neuron Neuron::operator=(const Neuron& neuron)
//{
//    if (this != &neuron) {
//        _weights = neuron._weights;
//        activationFunction = neuron.activationFunction;
//        _inputSize = neuron._inputSize;
//        _error = neuron._error;
//        _u = neuron._u;
//        _learningRate = neuron._learningRate;
//    }
//    return *this;
//}



//Json Neuron::ToJson() const
//{
//    Json json ={
//        {"bias", _weights[0] },
//        {"weights", _weights}
//    };
//
//    return json;
//}



//std::vector<double> Neuron::LoadWeightsFromJson(const Json& j)
//{
//    (*this)._weights = j.at("weights").get<std::vector<double>>();
//    return (*this)._weights;
//}

