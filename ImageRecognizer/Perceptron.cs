using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using Newtonsoft.Json;

namespace ImageRecognizer
{
    public class Perceptron
    {
        private double[][] Z;
        private double[][] activations;

        private double[][][] weights;
        private double[][] biases;
        
        private double[][][] dW;
        private double[][] dB;
        private double[][] dA;

        private ActivationFunctionType ActivationFunction = ActivationFunctionType.Sigmoid;
        
        public Perceptron(int[] config)
        {
            Init(config);
        }

        public int Recognize(byte[] image)
        {
            UploadImage(image);
            CalculateActivations();
            var result = ReadOutput();

            return result;
        }

        public int Recognize(MnistImageReader reader, int numberOfImages)
        {
            int matches = 0;
            for (int i = 0; i < numberOfImages; i++)
            {
                var (image, label) = reader.ReadImage();
                
                var result = Recognize(image);
                
                var match = result == label;
                Console.WriteLine($"{i+1}: match={match}, in={label}, out={result}");
                if (match) matches++;
            }

            Console.WriteLine($"RESULT: {((double)matches/(double)numberOfImages*100.0):F}% ({matches}/{numberOfImages})");
            
            return numberOfImages;
        }

        public void Train(MnistImageReader reader)
        {
            // Randomize network's parameters
            RandomizeWeights();
            RandomizeBiases();
            
            var batchSize = 1;
            var iterations = reader.ImagesCount / batchSize;
            var matches = 0;
            
            Console.WriteLine("Training started");
            var stopwatch = new Stopwatch();
            stopwatch.Restart();
            
            for (int i = 0; i < iterations; i++)
            {
                if (TrainBatch(batchSize, i, reader))
                {
                    matches++;
                }
                
                Console.Clear();
                Console.WriteLine($"{(double)(i+1)/iterations*100:F}%");
            }
            
            stopwatch.Stop();
            var elapsedTime = stopwatch.Elapsed;
            Console.WriteLine($"Trained on {reader.ImagesCount} images in {elapsedTime.Minutes}:{elapsedTime.Seconds}.{elapsedTime.Milliseconds}s.");
        }
        
        private bool TrainBatch(int batchSize, int iteration, MnistImageReader reader)
        {
            int i = 0, matches = 0;
            while (i < batchSize)
            {
                var (image, label) = reader.ReadImage();

                // Upload image into input layer
                UploadImage(image);

                // Create expected results vector
                var expectedResults = new double[activations[^1].Length];
                expectedResults[label] = 1;

                // 1. Recalculate all Zo and Sigma(Zo)
                CalculateActivations();

                // 2. Calculate all activations effect on cost function
                CalculateActivationsCostEffect(expectedResults);

                // 3. Accumulate gradient values
                AccumulateGradient();
                
                i++;
            }

            var output = ReadOutput();
            Console.WriteLine($"MATCH={output == reader.LastLabel}; in {reader.LastLabel}, out {output}");

            // 4. Apply gradient's delta
            ApplyGradient(batchSize, iteration);

            return output == reader.LastLabel;
        }

        // ------------------------------------------------------------------------------------------------

        private int ReadOutput()
        {
            var max = double.MinValue;
            var maxIndex = -1;
            var outputLayer = activations[^1];

            for (int i = 0; i < outputLayer.Length; i++)
            {
                if (outputLayer[i] > max)
                {
                    max = outputLayer[i];
                    maxIndex = i;
                }
            }

            return maxIndex;
        }

        public void CalculateActivations()
        {
            // If shape is 784, 16, 16, 10, starting with the 2nd layer it will be..
            for (int L = 1; L < Z.Length; L++)
            {
                // k = 0..15 - iterates over each neuron's activation of Lth layer
                for (int k = 0; k < Z[L].Length; k++)
                {
                    var weightedSum = 0.0;
                    
                    // i = 0..783, num of neurons on prev layer
                    // iterates over each weight of connection of a neuron on Lth layer with L-1st layer
                    for (int j = 0; j < Z[L-1].Length; j++)
                    {
                        weightedSum += weights[L-1][k][j] * activations[L-1][j];
                    }
                    
                    Z[L][k] = weightedSum + biases[L-1][k];
                    activations[L][k] = MathFunctions.ActivationFunction(Z[L][k], ActivationFunction);
                }
            }
        }

        private void CalculateActivationsCostEffect(double[] expectedResults)
        {
            for (int L = activations.Length - 1; L > 0; L--)
            {
                for (int k = 0; k < activations[L].Length; k++)
                {
                    // if current layer is output
                    if (L == activations.Length - 1)
                    {
                        dA[L][k] = MathFunctions.CostFunctionDerivative(activations[L][k], expectedResults[k]);
                    }
                    else
                    {
                        dA[L][k] = 0.0;
                        for (int j = 0; j < activations[L+1].Length; j++)
                        {
                            dA[L][k] += weights[L][j][k] * MathFunctions.ActivationFunctionDerivative(Z[L+1][j], ActivationFunction) * dA[L+1][j];
                        }
                    }
                }
            }
        }

        private void AccumulateGradient()
        {
            for (int L = 1; L < activations.Length; L++)
            {
                for (int k = 0; k < activations[L].Length; k++)
                {
                    dB[L - 1][k] += MathFunctions.ActivationFunctionDerivative(Z[L][k], ActivationFunction) * dA[L][k];
                    for (int j = 0; j < activations[L - 1].Length; j++)
                    {
                        dW[L - 1][k][j] += activations[L - 1][j] * dB[L - 1][k];
                    }
                }
            }
        }

        private void ApplyGradient(int batchSize, int iteration)
        {
            var initialRate = 0.1;
            var decay = 0.01;
            var learningRate = initialRate; // * (1 / (1 + decay * iteration));
            
            for (int L = 1; L < activations.Length; L++)
            {
                for (int k = 0; k < activations[L].Length; k++)
                {
                    biases[L - 1][k] += -learningRate * (dB[L - 1][k] / batchSize);
                    
                    // Reset after application
                    dB[L - 1][k] = 0;
                    for (int j = 0; j < activations[L - 1].Length; j++)
                    {
                        weights[L - 1][k][j] += -learningRate * (dW[L - 1][k][j] / batchSize);
                        
                        // Reset after application
                        dW[L - 1][k][j] = 0;
                    }
                }
            }
        }

        public void UploadImage(byte[] image)
        {
            var inputLayer = activations[0];
            for (int i = 0; i < inputLayer.Length; i++)
            {
                // Squeeze pixel byte to range of 0..1 
                inputLayer[i] = image[i]/255.0; 
            }
        }

        private void RandomizeWeights()
        {
            var random = new Random();
            
            for (int L = 0; L < weights.Length; L++)
            {
                for (int k = 0; k < weights[L].Length; k++)
                {
                    for (int j = 0; j < weights[L][k].Length; j++)
                    {
                        weights[L][k][j] = random.NextDouble() * 1 - .5;
                    }
                }
            }
        }

        private void RandomizeBiases()
        {
            var random = new Random();

            for (int L = 0; L < biases.Length; L++)
            {
                for (int k = 0; k < biases[L].Length; k++)
                {
                    biases[L][k] = random.NextDouble() * 1 - .5;
                }
            }
        }

        public void Export(string path)
        {
            var network = new NetworkSettings
            {
                Biases = biases,
                Weights = weights
            };

            var json = JsonConvert.SerializeObject(network);
            using var fs = new FileStream(path, FileMode.Create);
            var bytes = Encoding.Default.GetBytes(json);
            fs.Write(bytes);
        }

        public void Import(string path)
        {
            var bytes = File.ReadAllBytes(path);
            var json = Encoding.Default.GetString(bytes);
            var network = JsonConvert.DeserializeObject<NetworkSettings>(json);

            biases = network.Biases;
            weights = network.Weights;
        }

        /// <summary>
        /// Build activations, weights, biases and gradient with default values
        /// </summary>
        /// <param name="shape">Shape of the network</param>
        private void Init(int[] shape)
        {
            int layersNum = shape.Length;
                        
            Z = new double[layersNum][]; 
            activations = new double[layersNum][];

            weights = new double[layersNum - 1][][];
            biases = new double[layersNum - 1][];
            
            dW = new double[layersNum - 1][][];
            dB = new double[layersNum - 1][];
            dA = new double[layersNum][];

            for (int i = 0; i < shape.Length; i++)
            {
                var layerCapacity = shape[i];
                
                Z[i] = new double[layerCapacity];
                activations[i] = new double[layerCapacity];
                dA[i] = new double[layerCapacity];
                
                if (i < shape.Length - 1)
                {
                    weights[i] = new double[shape[i+1]][];
                    biases[i] = new double[shape[i+1]];
                    
                    dW[i] = new double[shape[i+1]][];
                    dB[i] = new double[shape[i+1]];

                    for (int j = 0; j < shape[i+1]; j++)
                    {
                        weights[i][j] = new double[layerCapacity];
                        dW[i][j] = new double[layerCapacity];
                    }
                }
            }
        }
    }

    public class NetworkSettings
    {
        public double[][][] Weights;
        public double[][] Biases;
    }
}