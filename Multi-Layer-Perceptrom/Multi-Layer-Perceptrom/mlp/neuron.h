#pragma once

#include <nlohmann/json.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "../utils/basic-includes.h"
#include "../utils/utils.h"
#include "activation-functions.h"
#include "lost-function.h"

using Json = nlohmann::json;

class Layer;


class Neuron {

	private:
	//--- atributos fundamentais do meuronios
		double _learningRate;
		Eigen::MatrixXd _weights;
		IActivationFunction* activationFunction;
		ILostFunction* _lostFunction;

		double _delta;     // (dL/dU) -> loss partial with respect to weighted sum
		Eigen::MatrixXd _lossGradient;

	//--- variaveis de armazenamento
		size_t _inputSize;                                              // tamanho do vetor de entrada
		double _error;	                                                // erro do do valor de saida
		double _u;		                                                // weighted sum

		

	public:
	//--- construtor
		Neuron(size_t inputSize, IActivationFunction* actFun = new Sigmoid(), double leraningRate = 0.01, ILostFunction* lostFunc = nullptr, size_t outputSize = 0);
		~Neuron();


	//--- pricipais methods
		double CalculateOutput(Eigen::MatrixXd inputs);

		void CalculateLossGradient(double correctValue, double predictedValue, Eigen::MatrixXd input);
		void CalculateLossGradient(double LossPartialWithRespectToActivation, Eigen::MatrixXd input);
		double LossPartialWithRespectToInput(size_t inputIndex);
		void UpdateWeights();


	//--- get e sets
		void XavierWeightInitialization(size_t inputSize, size_t outputSize);
		double& operator[](int weightIndex);
		//Neuron operator=(const Neuron& neuron);

		//Json ToJson() const;
		//std::vector<double> LoadWeightsFromJson(const Json& j);


		enum class Attribute { 
			BIAS, ERROR, U, GRADIENT_DL_DU, LEARNING_RATE,      // double types
			WEIGHTS, OUTPUT, 									// std::vector<double> type
			ACTIVATION_FUNC,									// IActivationFunction* TYPE
			LOST_FUNC											// ILostFunction* 
		};
		template <Attribute attrib> const auto Get() const;
		template <Attribute attrib, typename T> void Set(T value);

	friend class Layer;
};




template<Neuron::Attribute attrib>
const auto Neuron::Get() const
{
	if constexpr (attrib == Neuron::Attribute::ACTIVATION_FUNC) {
		return activationFunction;
	}
	else if constexpr (attrib == Neuron::Attribute::LOST_FUNC) {
		return _lostFunction;
	}
	else if constexpr (attrib == Attribute::WEIGHTS) {
		return _weights;
	} 
	else if constexpr (attrib == Attribute::BIAS) {
		return _weights(0,0);
	} 
	else if constexpr (attrib == Attribute::ERROR) {
		return _error;
	} 
	else if constexpr (attrib == Attribute::U) {
		return _u;
	}
	else {
		assert(false  &&  "Unsupported attribute");
	}
}



template<Neuron::Attribute attrib, typename T>
void Neuron::Set(T value)
{
	if constexpr (attrib == Neuron::Attribute::ACTIVATION_FUNC) {
		static_assert( std::is_same_v<T, IActivationFunction*>  &&  "wrong type");
		activationFunction  =  value;
	}
	else if constexpr (attrib == Neuron::Attribute::LOST_FUNC) {
		static_assert(std::is_same_v<T, ILostFunction*>  &&  "wrong type");
		_lostFunction  =  value;
	}
	else if constexpr (attrib == Neuron::Attribute::LEARNING_RATE) {
		static_assert(std::is_same_v<T, double>  &&  "wrong type");
		_learningRate  =  value;
	}
	else {
		assert(false && "Not settable attribute");
	}
}
