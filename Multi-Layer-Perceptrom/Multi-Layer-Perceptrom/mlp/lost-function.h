#pragma once

class ILostFunction {
	public:
	virtual double f(double predicted, double correct) = 0;
	virtual double df(double predicted, double correct) = 0;
};




//--- mean absolute error
class MAE : public ILostFunction {
	public:
	virtual double f(double predicted, double correct) override;
	virtual double df(double predicted, double correct) override;
};



//--- mean square error
class MSE : public ILostFunction {
	public:
	virtual double f(double predicted, double correct) override;
	virtual double df(double predicted, double correct) override;
};



//--- Root Mean Square error
class RMSE : public ILostFunction {
	public:
	virtual double f(double predicted, double correct) override;
	virtual double df(double predicted, double correct) override;
};



//--- Cross Entropy
class CrossEntropy : public ILostFunction {
	public:
	virtual double f(double predicted, double correct) override;
	virtual double df(double predicted, double correct) override;
};
