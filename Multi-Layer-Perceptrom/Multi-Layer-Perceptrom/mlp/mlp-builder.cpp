#include "mlp-builder.h"

MlpBuilder::MlpBuilder()
{
	inputSize = 0;
	_mlp = MLP();
	_mlp._outFile = "";
	_mlp.HowToUpdateLeraningRate  =  [](size_t e, double a, double currentRate){ return currentRate; };
	_mlp.WhenToUpdateLeraningRate = [](size_t e, double a){ return false; };
}



MLP MlpBuilder::Build()
{
	
	auto& lastLayer  =  _mlp._layers[_mlp._layers.size()-1];
	size_t neuronsInLastLayer = lastLayer.Get<Layer::Attribute::NUMBER_OF_NEURONS>();
	_mlp._accumulatedGradients  =  std::vector<double>(neuronsInLastLayer, 0.0);

	return _mlp;
}


MlpBuilder MlpBuilder::InputSize(size_t size)
{
	inputSize = size;
	_mlp._inputSize = size;

	return (*this);
}


MlpBuilder MlpBuilder::Architecture(std::vector<size_t> neuronsInLayer)
{
	assert( inputSize == 0 && "DEFINE inputsize FISRT" );


	for (size_t i = 0; i < neuronsInLayer.size(); i++) {
		_mlp._layers.push_back( Layer(inputSize, neuronsInLayer[i]) );
		inputSize  =  neuronsInLayer[i];
	}

	return (*this);
}


MlpBuilder MlpBuilder::Architecture(std::vector<LayerSignature> layerSignature) 
{
	assert( inputSize > 0 && "DEFINE inputsize FISRT" );

	for (size_t i = 0; i < layerSignature.size(); i++) { // from [0] to [n-1]

		size_t nextLayerNeuronQnt = layerSignature[i]._qntNeurons; // =  layerSignature[i+1]._qntNeurons;
		if (i+1 < layerSignature.size()) { nextLayerNeuronQnt = layerSignature[i+1]._qntNeurons; }

		size_t qntNeuron  =  layerSignature[i]._qntNeurons;
		IActivationFunction* actFunc  =  layerSignature[i]._activationFunction;
		ILostFunction* lostFunc  =  layerSignature[i]._lostFunction;
		double learningRate  =  layerSignature[i]._learningRate;

		_mlp._layers.push_back( Layer(inputSize, qntNeuron, actFunc, learningRate, lostFunc, nextLayerNeuronQnt) );
		inputSize  =  qntNeuron;
	}

	return (*this);
}



MlpBuilder MlpBuilder::LostFunction(ILostFunction* lostFunction)
{
	assert( _mlp._layers.size() > 0 && "DEFINE LAYER ARCHITECTURE FIRST");

	size_t lastLayerIndex = _mlp._layers.size()-1;
	auto& lastLayer  =  _mlp._layers[lastLayerIndex];
	lastLayer.Set<Layer::Attribute::LOSS_FUNC, ILostFunction*>(lostFunction);

	//size_t neuronSize = lastLayer.Get<Layer::Attribute::NUMBER_OF_NEURONS>();
	//for (size_t i = 0; i < neuronSize; i++) {
	//	//lastLayer[i].SetLostFunction( lostFunction );
	//	lastLayer[i].Set<Neuron::Attribute::LOST_FUNC, ILostFunction*>( lostFunction );
	//}

	return (*this);
}



MlpBuilder MlpBuilder::MaxEpochs(size_t epochs)
{
	_mlp._maxEpochs = epochs;

	return (*this);
}

MlpBuilder MlpBuilder::AcceptableAccuracy(double accuracy)
{
	assert( 0.0 < accuracy && accuracy < 1.0  &&  "ACCURACY MUST BE BETWEEN 0 AND 1" );

	_mlp._acceptableAccuracy  =  accuracy;

	return (*this);
}

MlpBuilder MlpBuilder::ParseLabelToVector(std::function<std::vector<double>(size_t)> CallBack)
{
	_mlp.ParseLabelToVector = CallBack;
	return (*this);
}


MlpBuilder MlpBuilder::SaveOn(std::string outFile)
{
	_mlp._outFile = outFile;
	return (*this);
}



MlpBuilder MlpBuilder::LoadArchitectureFromJson(std::string file)
{
	Json json;
	std::ifstream jsonFile(file);
	assert( jsonFile.is_open()  &&  "[ERROR]: file not Found");
	jsonFile >> json;
	jsonFile.close();

	_mlp._inputSize  =  json.at("inputSize").get<size_t>();


	for (const auto& layerJson : json.at("layers")) {
		size_t numberOfNeurons  =  layerJson.at("layer").at("neurons").size();
		Layer layer = Layer(0, numberOfNeurons);
		layer.LoadWeightsFromJson( layerJson );

		_mlp._layers.push_back( layer );
	}


	return (*this);
}

MlpBuilder MlpBuilder::WhenToUpdateLearningRate(std::function<bool(size_t, double)> Conddition)
{
	_mlp.WhenToUpdateLeraningRate  =  Conddition;
	return (*this);
}

MlpBuilder MlpBuilder::HowToUpdateLearningRate(std::function<double(size_t, double, double)> func)
{
	_mlp.HowToUpdateLeraningRate  =  func;
	return (*this);
}



