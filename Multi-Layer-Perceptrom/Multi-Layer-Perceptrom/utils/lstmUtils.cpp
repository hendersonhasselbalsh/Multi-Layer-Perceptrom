#include "lstmUtils.h"



void Utils::PointwiseAdd(std::vector<double>* v1, std::vector<double>* v2, std::vector<double>* result)
{
	std::transform(v1->begin(), v1->end(), v2->begin(), result->begin(), std::plus<double>());
}



void Utils::PointwiseMult(std::vector<double>* v1, std::vector<double>* v2, std::vector<double>* result)
{
	std::transform(v1->begin(), v1->end(), v2->begin(), result->begin(), std::multiplies<double>());
}



std::vector<double> Utils::PointwiseTanh(std::vector<double>& v1)
{
	std::vector<double> result = std::vector<double>(v1.size(), 0.0);

	size_t i = 0;
	for (auto& x : v1) {
		result[i]  =  std::tanh( x );
		i++;
	}

	return result;
}



