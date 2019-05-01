using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Microsoft.ML;

namespace VeryCoolProject
{
    public partial class Form1 : Form
    {
        private PredictionEngine<ImageData, ImageDataPrediction> _engine;
        public Form1()
        {
            InitializeComponent();
        }

        private void Button1_Click(object sender, EventArgs e)
        {
            if (openFileDialog1.ShowDialog() == DialogResult.OK &&
                _engine != null &&
                File.Exists(openFileDialog1.FileName) &&
                openFileDialog1.FileName.EndsWith(".jpg"))
            {
                Cursor = Cursors.WaitCursor;
                pictureBox1.ImageLocation = openFileDialog1.FileName;
                var transformation = _engine.Predict(new ImageData { Location = openFileDialog1.FileName });
                label1.Text = transformation.Prediction;
                Cursor = Cursors.Default;
            }
        }

        private void Button2_Click(object sender, EventArgs e)
        {
            if (openFileDialog1.ShowDialog() == DialogResult.OK &&
                File.Exists(openFileDialog1.FileName) &&
                openFileDialog1.FileName.EndsWith(".zip"))
            {
                Cursor = Cursors.WaitCursor;
                var context = new MLContext();
                var model = context.Model.Load(openFileDialog1.FileName, out var schema);
                _engine = context.Model.CreatePredictionEngine<ImageData, ImageDataPrediction>(model);
                Cursor = Cursors.Default;
            }
        }
    }
}
