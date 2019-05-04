using Plugin.Media.Abstractions;
using Plugin.Media;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xam.Plugins.OnDeviceCustomVision;
using Xamarin.Forms;

namespace TacosML_Xamarin
{
    // Learn more about making custom code visible in the Xamarin.Forms previewer
    // by visiting https://aka.ms/xamarinforms-previewer
    [DesignTimeVisible(true)]
    public partial class MainPage : ContentPage
    {
        public MainPage()
        {
            InitializeComponent();
        }

        private async void Button1_ClickedAsync(object sender, EventArgs e)
        {
            try
            {

                await CrossMedia.Current.Initialize();

                MediaFile file = await CrossMedia.Current.PickPhotoAsync(new PickMediaOptions{ PhotoSize = PhotoSize.Medium });
                Stream contents = file.GetStream();

                System.Console.WriteLine("File path: " + file.Path);

                var label = await CrossImageClassifier.Current.ClassifyImage(contents);

            }
            catch (Exception ex)
            {
                System.Console.WriteLine("Exception choosing file: " + ex.ToString());
            }
        }
    }
}
