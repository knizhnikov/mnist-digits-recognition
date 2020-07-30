using System;
using System.Security.Cryptography;

namespace ImageRecognizer
{
    public static class MathFunctions
    {
        // Ro(Z)
        public static double ActivationFunction(double x, ActivationFunctionType functionType)
        {
            switch (functionType)
            {
                case ActivationFunctionType.Sigmoid:
                    return 1 / (1 + Math.Exp(-x));
                case ActivationFunctionType.Softplus:
                    return Math.Log(1 + Math.Exp(x));
                case ActivationFunctionType.ReLU:
                    return x < 0 ? 0 : x;
            }

            throw new ArgumentException("Not supported function type");
        }

        // Ro'(Z)
        public static double ActivationFunctionDerivative(double x, ActivationFunctionType functionType)
        {
            switch (functionType)
            {
                case ActivationFunctionType.Sigmoid:
                    return ActivationFunction(x, ActivationFunctionType.Sigmoid) * (1 - ActivationFunction(x, ActivationFunctionType.Sigmoid));
                    //return Math.Exp(-x) / Math.Pow(1 + Math.Exp(-x), 2);
                case ActivationFunctionType.ReLU:
                    return x > 0 ? 1 : 0;
            }
            
            throw new ArgumentException("Not supported function type");
        }
        
        // Co
        public static double CostFunction(double actualResult, double expectedResult)
        {
            return Math.Pow(actualResult - expectedResult, 2);
        }
        
        // Co' = dCo/da0
        public static double CostFunctionDerivative(double actualResult, double expectedResult)
        {
            return 2 * (actualResult - expectedResult);
        }
        
        
    }
    
    public enum ActivationFunctionType
    {
        Sigmoid,
        ReLU,
        Softplus
    }
}