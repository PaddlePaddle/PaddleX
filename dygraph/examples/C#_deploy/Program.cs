using System;
using System.Runtime.InteropServices;
using System.Drawing.Imaging;
using System.Drawing;

namespace ConsoleApp2
{
    class Program
    {
        [DllImport("model_infer.dll", EntryPoint = "InitModel")]
        public static extern void InitModel(string model_type, string model_filename, string params_filename, string cfg_file);

        [DllImport("model_infer.dll", EntryPoint = "ModelPredict")]
        public static extern void ModelPredict(byte[] img, int W, int H, int C, IntPtr output, int[] BoxesNum, ref byte label);

        [DllImport("model_infer.dll", EntryPoint = "DestructModel")]
        public static extern void DestructModel();


        static void Main(string[] args)
        {
            string imgfile = "E:\\PaddleX_deploy\\PaddleX\\dygraph\\deploy\\cpp\\out\\paddle_deploy\\1.png";
            string model_type = "det";
            string model_filename = "E:\\PaddleX_deploy\\PaddleX\\dygraph\\deploy\\cpp\\out\\paddle_deploy\\yolov3_darknet53_270e_coco1\\model.pdmodel";
            string params_filename = "E:\\PaddleX_deploy\\PaddleX\\dygraph\\deploy\\cpp\\out\\paddle_deploy\\yolov3_darknet53_270e_coco1\\model.pdiparams";
            string cfg_file = "E:\\PaddleX_deploy\\PaddleX\\dygraph\\deploy\\cpp\\out\\paddle_deploy\\yolov3_darknet53_270e_coco1\\infer_cfg.yml";


            InitModel(model_type, model_filename, params_filename, cfg_file);


            Bitmap bmp = new Bitmap(imgfile);
            byte[] inputData = GetBGRValues(bmp, out int stride);

            float[] resultlist = new float[600];
            IntPtr results = FloatToIntptr(resultlist);
            int[] boxesInfo = new int[1];
            Byte[] labellist = new Byte[1000];    //新建字节数组

            //第四个参数为输入图像的通道数
            ModelPredict(inputData, bmp.Width, bmp.Height, 4, results, boxesInfo, ref labellist[0]);
            string strGet = System.Text.Encoding.Default.GetString(labellist, 0, labellist.Length);    //将字节数组转换为字符串
            Console.WriteLine("labellist: {0}", strGet);
            for (int i = 0; i < boxesInfo[0]; i++)
            {
                int labelindex = Convert.ToInt32(resultlist[i * 6 + 0]);
                float score = resultlist[i * 6 + 1];
                float left = resultlist[i * 6 + 2];
                float top = resultlist[i * 6 + 3];
                float width = resultlist[i * 6 + 4];
                float height = resultlist[i * 6 + 5];
                Console.WriteLine("score: {0}", score);
                Console.WriteLine("labelindex: {0}", labelindex);
                Console.WriteLine("boxe: {0} {1} {2} {3}", left, top, width, height);
            }

            DestructModel();
        }

        // 将Btimap类转换为byte[]类函数
        public static byte[] GetBGRValues(Bitmap bmp, out int stride)
        {
            var rect = new Rectangle(0, 0, bmp.Width, bmp.Height);
            var bmpData = bmp.LockBits(rect, ImageLockMode.ReadOnly, bmp.PixelFormat);
            stride = bmpData.Stride;
            var rowBytes = bmpData.Width * Image.GetPixelFormatSize(bmp.PixelFormat) / 8;
            var imgBytes = bmp.Height * rowBytes;
            byte[] rgbValues = new byte[imgBytes];
            IntPtr ptr = bmpData.Scan0;
            for (var i = 0; i < bmp.Height; i++)
            {
                Marshal.Copy(ptr, rgbValues, i * rowBytes, rowBytes);
                ptr += bmpData.Stride;
            }
            bmp.UnlockBits(bmpData);
            return rgbValues;
        }

        public static IntPtr FloatToIntptr(float[] bytes)
        {
            GCHandle hObject = GCHandle.Alloc(bytes, GCHandleType.Pinned);
            return hObject.AddrOfPinnedObject();
        }


    }
}
