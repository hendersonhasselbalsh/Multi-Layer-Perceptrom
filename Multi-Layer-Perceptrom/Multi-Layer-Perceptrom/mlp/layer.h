#pragma once

#include <limits>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include "activation-functions.h"
#include "lost-function.h"
#include "../utils/utils.h"
//#include "neuron.h"
using Json = nlohmann::json;

class MLP;
class LSTM;


class Layer {

	private:
	//--- atrubutos principais da classe
		Eigen::MatrixXd _weights;					// _weights[ neuron_Index ][ weight_Index ]
		Eigen::MatrixXd _outputs;
		Eigen::MatrixXd _receivedInput;
	
	//--- atributos dos neuronios da classe
		IActivationFunction* _activationFunc;
		ILostFunction* _lostFunction;
		double _learningRate;

	//--- variaveis de armazenamento auxiliares
		size_t _inputSize;
	
	//--- neuron attributes
		Eigen::MatrixXd _weightedSums;




	public:
	//--- construtor
		Layer(size_t inputSize, size_t neuronQuantity, IActivationFunction* actFun = new Sigmoid(), double neuronLerningRate = 0.01, ILostFunction* _lostFunc = nullptr, size_t nextLayerNeuronQnt = 0);
		~Layer();


	//--- methods
		std::vector<double> CalculateLayerOutputs(std::vector<double> inputs);

		Eigen::MatrixXd UpdateLastLayerWeight(std::vector<double> predictedValues, std::vector<double> correctValues);
		Eigen::MatrixXd UpdateHiddenLayerWeight(Eigen::MatrixXd dLoss_dActivation);


	//--- foward
		Eigen::MatrixXd WeightedSum(Eigen::MatrixXd& weights, Eigen::MatrixXd& input);
		Eigen::MatrixXd Activation(Eigen::MatrixXd& weightSums);

	//--- chain's rule
		Eigen::MatrixXd LossPartialWithRespectToActivation(std::vector<double> predictedValues, std::vector<double> correctValues);
		Eigen::MatrixXd LossPartialWithRespectToWeightedSum(Eigen::MatrixXd& dLoss_dActivation);
		Eigen::MatrixXd LossPartialWithRespectToWeight(Eigen::MatrixXd& dLoss_dWeightedSun);
		Eigen::MatrixXd LossPartialWithRespectToInput(Eigen::MatrixXd& dLoss_dWeightedSun);



	//--- metodos de acesso (get e set)
		void XavierWeightInitialization(size_t inputSize, size_t outputSize);

		 Json ToJson() const;
		 Layer LoadWeightsFromJson(const Json& j);

		 enum class Attribute { 
			 INPUT_SIZE, OUTPUT_SIZE, NUMBER_OF_NEURONS, 								                 // TYPE: size_t
			 ALL_NEURONS, 																                 // TYPE: std::vector<Neuron>
			 LAYER_OUTPUTS, LAYER_ERRORS, ALL_NEURONS_GRADIENTS, ACCUMULATED_OUTPUTS, RECEIVED_INPUT,	 // TYPE: std::vector<double>
			 LEARNING_RATE, 																			 // TYPE: double
			 LOSS_FUNC
		 };
		 template <Layer::Attribute attrib> const auto Get() const;
		 template <Layer::Attribute attrib, typename T> void Set(T value);


	friend class MLP;
	friend class LSTM;
};




template<Layer::Attribute attrib>
inline const auto Layer::Get() const
{
	if constexpr (attrib == Layer::Attribute::INPUT_SIZE) {
		return _inputSize;
	}
	else if constexpr (attrib == Layer::Attribute::OUTPUT_SIZE) {
		return _outputs.size();
	}
	else if constexpr (attrib == Layer::Attribute::NUMBER_OF_NEURONS) {
		return (size_t)_weights.rows();
		//return _neurons.size();
	}
	/*else if constexpr (attrib == Layer::Attribute::ALL_NEURONS) {
		return _neurons;
	}*/
	else if constexpr (attrib == Layer::Attribute::LAYER_OUTPUTS) {
		return _outputs;
	}
	/*else if constexpr (attrib == Layer::Attribute::LAYER_ERRORS) {
		std::vector<double> errors;
		for (auto& neuron : _neurons) { errors.push_back( neuron.Get<Neuron::Attribute::ERROR>() ); }
		return errors;
	}*/
	else if constexpr (attrib == Layer::Attribute::RECEIVED_INPUT) {
		return _receivedInput;
	}
	else {
		assert(false && "cant get this attribute");
	}
}


template<Layer::Attribute attrib, typename T>
inline void Layer::Set(T value)
{
	if constexpr (attrib == Layer::Attribute::LEARNING_RATE) {
		static_assert( std::is_same_v<T, double>  &&  "[ERROR]: wrong type");
		_learningRate  =  value;
	}
	/*else if constexpr (attrib == Layer::Attribute::ALL_NEURONS_GRADIENTS) {
		static_assert( std::is_same_v<T, std::vector<double>>  &&  "[ERROR]: wrong type");
		size_t index = 0;
		for (auto& neuron : _neurons) {
			neuron.Set<Neuron::Attribute::GRADIENT_DL_DU, double>(value[index++]);
		}
	}*/
	else if constexpr (attrib == Layer::Attribute::LOSS_FUNC) {
		static_assert(std::is_same_v<T, ILostFunction*>  &&  "wrong type");
		_lostFunction  =  value;
	}
	else {
		assert(false  &&  "[ERROR]: not settable attribute");
	}
}
