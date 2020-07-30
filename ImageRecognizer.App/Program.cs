using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace ImageRecognizer.App
{
    class Program
    {
        static void Main(string[] args)
        {
            var trainingSetPath = @"/Users/knizhnikov/Documents/Dev/Projects/Perceptron/Perceptron/ImageRecognizer.App/train-images-idx3-ubyte";
            var trainingSetLabelsPath =
                @"/Users/knizhnikov/Documents/Dev/Projects/Perceptron/Perceptron/ImageRecognizer.App/train-labels-idx1-ubyte";

            using var reader = new MnistImageReader(trainingSetPath, trainingSetLabelsPath);
            
            SaveImages(10, reader);
            
            var perceptron = new Perceptron(new []{ reader.Columns * reader.Rows, 100, 100, 10 });
            // perceptron.Train(reader);
            //
            // //perceptron.Export(@"/Users/knizhnikov/Documents/Dev/Projects/Perceptron/Perceptron/ImageRecognizer.App/network");
            //
            // var testSetPath =
            //     @"/Users/knizhnikov/Documents/Dev/Projects/Perceptron/Perceptron/ImageRecognizer.App/t10k-images-idx3-ubyte";
            // var testSetLabelsPath =
            //     @"/Users/knizhnikov/Documents/Dev/Projects/Perceptron/Perceptron/ImageRecognizer.App/t10k-labels-idx1-ubyte";
            //
            // using var testReader = new MnistImageReader(testSetPath, testSetLabelsPath);
            // perceptron.Recognize(testReader, 10000);
        }

        public static void SaveImages(int num, MnistImageReader reader)
        {
            for (int k = 0; k < num; k++)
            {
                var (image, label) = reader.ReadImage();
                var bitmap = new Bitmap(28, 28);

                for (int i = 0; i < image.Length; i++)
                {
                    bitmap.SetPixel(i/28, i%28, Color.FromArgb(image[i], image[i], image[i]));
                }
                
                bitmap.Save($"/Users/knizhnikov/Documents/Dev/Projects/Perceptron/Perceptron/ImageRecognizer.App/images/{k}.jpeg", ImageFormat.Jpeg);
            }
        }
    }
}