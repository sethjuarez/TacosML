using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace VeryCoolProject
{
    public class ImageData
    {
        public string Location { get; set; }
        public string Label { get; set; }
    }

    public class ImageDataPrediction
    {
        public string Location { get; set; }
        public float[] Score { get; set; }
        public string Prediction { get; set; }
    }
}
