using System;
using System.IO;

namespace ImageRecognizer
{
    public class MnistImageReader : IDisposable
    {
        private readonly FileStream _imagesFileStream;
        private readonly FileStream _labelsFileStream;

        private readonly int _imagesMagic;
        private readonly int _imagesCount;
        private readonly int _rows;
        private readonly int _columns;
        
        private readonly int _labelsMagic;
        private readonly int _labelsCount;

        public int ImagesCount => _imagesCount;
        public int Rows => _rows;
        public int Columns => _columns;
        public int Position { get; set; }
        public int LastLabel { get; set; }
        
        public MnistImageReader(string imagesPath, string labelsPath)
        {
            // Get images info
            _imagesFileStream = new FileStream(imagesPath, FileMode.Open);
            
            _imagesMagic = ReadIntFromStream(_imagesFileStream);
            _imagesCount = ReadIntFromStream(_imagesFileStream);
            _rows = ReadIntFromStream(_imagesFileStream);
            _columns = ReadIntFromStream(_imagesFileStream);
            
            // Get labels info
            _labelsFileStream = new FileStream(labelsPath, FileMode.Open);
            
            _labelsMagic = ReadIntFromStream(_labelsFileStream);
            _labelsCount = ReadIntFromStream(_labelsFileStream);
            
            if (_labelsCount != _imagesCount) throw new FileLoadException("Images and labels file mismatch");
        }

        public void Dispose()
        {
            _imagesFileStream.Dispose();
            _labelsFileStream.Dispose();
        }

        public (byte[], int) ReadImage()
        {
            var buffer = new byte[_rows * _columns];
            _imagesFileStream.Read(buffer, 0, buffer.Length);
            var label = _labelsFileStream.ReadByte();

            Position++;
            LastLabel = label;
            
            return (buffer, label);
        }

        private int ReadIntFromStream(Stream stream)
        {
            var buffer = new byte[4];
            stream.Read(buffer, 0, 4);
            
            Array.Reverse(buffer);
            return BitConverter.ToInt32(buffer, 0);
        }
    }
}