using System.Collections.Generic;
using System.Drawing;

namespace Neural_Network
{
    public class PictureConverter //only for windows
    {
        public int boundary { get; set; } = 128;
        public int Height { get; set; }
        public int Width { get; set; }
        public List<double> Convert(string path)
        {
            var result = new List<double>();

            var image = new Bitmap(path);

            Height = image.Height;
            Width = image.Width;

            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    var pixel = image.GetPixel(x, y);
                    var value = Brightness(pixel);
                    result.Add(value);
                }
            }

            return result;
        }

        private double Brightness(Color color)
        {         
            var result = 0.299 * color.R + 0.587 * color.G + 0.114 * color.B;
            return result < boundary ? 0 : 1;
        }

        public void Save(string path,List<double> pixels) //for check image
        {
            var image = new Bitmap(Width, Height);
            for (int y = 0; y < Height; y++)
            {
                for (int x = 0; x < Width; x++)
                {
                    var color = pixels[y * Width + x] == 1 ? Color.White : Color.Black;

                    image.SetPixel(x, y, color);
                }
            }
             
            image.Save(path);
        }
    }
}
