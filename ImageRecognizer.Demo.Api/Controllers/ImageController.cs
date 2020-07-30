using System;
using System.Buffers.Text;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;

namespace ImageRecognizer.Demo.Api.Controllers
{
    public class ImageController : Controller
    {
        private Perceptron _perceptron;

        public ImageController()
        {
            _perceptron = new Perceptron(new[] {784, 100, 100, 10});
            _perceptron.Import("AppData/network");
        }
        
        [HttpPost("recognize")]
        public ActionResult<int> Recognize([FromBody]string base64Image)
        {
            var image = Convert.FromBase64String(base64Image);
            var result = _perceptron.Recognize(image);
            return Ok(result);
        }
    }

    public class ImageDataRequest
    {
        public byte[] Data { get; set; }
    }
}