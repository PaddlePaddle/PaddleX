using System;
using System.IO;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Runtime.InteropServices;
using System.Drawing.Imaging;

using OpenCvSharp;


namespace WinFormsApp_final
{
    public partial class Form1 : Form
    {
        /**********************************************************************/
        /*****************          1.Reasoning DLL import implementation          ****************/
        /**********************************************************************/
        // Load inference correlation methods
        [DllImport("model_infer.dll", EntryPoint = "InitModel")] // Model unified initialization method: need yml、pdmodel、pdiparams
        public static extern void InitModel(string model_type, string model_filename, string params_filename, string cfg_file, bool use_gpu, int gpu_id, ref byte paddlex_model_type);

        [DllImport("model_infer.dll", EntryPoint = "Det_ModelPredict")]  // PaddleDetection Model reasoning method
        public static extern void Det_ModelPredict(byte[] img, int W, int H, int C, IntPtr output, int[] BoxesNum, ref byte label);

        [DllImport("model_infer.dll", EntryPoint = "Seg_ModelPredict")]  // PaddleSeg Model reasoning method
        public static extern void Seg_ModelPredict(byte[] img, int W, int H, int C, ref byte output);

        [DllImport("model_infer.dll", EntryPoint = "Cls_ModelPredict")]  // PaddleClas Model reasoning method
        public static extern void Cls_ModelPredict(byte[] img, int W, int H, int C, ref float score, ref byte category, ref int category_id);

        [DllImport("model_infer.dll", EntryPoint = "Mask_ModelPredict")]  // Paddlex-MaskRCNN Model reasoning method
        public static extern void Mask_ModelPredict(byte[] img, int W, int H, int C, IntPtr output, ref byte Mask_output, int[] BoxesNum, ref byte label);

        [DllImport("model_infer.dll", EntryPoint = "DestructModel")]  // Segmentation, detection, identification model destruction method
        public static extern void DestructModel();


        /**********************************************************************/
        /******************         2.Statement of control parameters          *****************/
        /**********************************************************************/

        string imgfile = null; // Single image path
        List<string> imgfiles = new List<string>(); // Multiple image paths
        string videofile = null; // Reasoning the video path
        bool use_gpu = false;  // Whether to use GPU
        int gpu_id = 0;
        float det_threshold = 0.5F;
        string model_type = "det";
        string model_filename = null; // *.pdmodel
        string params_filename = null;  // *.pdiparams
        string cfg_file = null;  // *.yml

        byte[] paddlex_model_type = new byte[10];  // det/seg/clas + \0
        bool paddlex_doing = false;
        string[] img_type = {"jpg", "png", "JPEG", "jpeg"};

        static int has_model_init = 0;
        static int is_infer = 0;
        static int isBreakInfer = 0;
        static int infer_one_img_flag = 0;
        static int infer_many_img_flag = 0;
        static int infer_video_img_flag = 0;
        int continue_infer_delay = 50;


        /**********************************************************************/
        /**************         3.Form loading and closing implementation          ***************/
        /**********************************************************************/
        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            comboBox1.SelectedIndex = 0;
            comboBox2.SelectedIndex = 0;
            comboBox3.SelectedIndex = 5;
            numericUpDown1.Value = 50;

            label7.Text = "0.00";
            textBox1.Text = "0";
        }

        private void Form1_FromClosing(object sender, EventArgs e)
        {
            while (is_infer != 0)
            {
                isBreakInfer = 1;
            }

            if (has_model_init == 1)
            {
                DestructModel();
            }
        }


        /**********************************************************************/
        /*****************          4.Detail parameter option implementation          ***************/
        /**********************************************************************/
        int comboBox_clicked = 0;
        int comboBox1_last_index = 0;

        // Select run environment - GPU or CPU
        private void comboBox1_SelectedIndexChanged(object sender, EventArgs e)
        {

            if (is_infer == 1 && comboBox_clicked == 0)
            {
                MessageBox.Show("正在推理中，请推理完成后再选择运行环境重新初始化!", "提示", MessageBoxButtons.OK, MessageBoxIcon.Error);
                comboBox_clicked = 1;
                comboBox1.SelectedIndex = comboBox1_last_index;
                return;
            }

            if (has_model_init == 1 && comboBox_clicked == 0)
            {
                MessageBox.Show("模型已初始化，请销毁模型后再进行运行环境选择!\n(CPU,GPU)", "提示", MessageBoxButtons.OK, MessageBoxIcon.Error);
                comboBox_clicked = 1;
                comboBox1.SelectedIndex = comboBox1_last_index;
                return;
            }

            if (comboBox1.SelectedItem.ToString() == "GPU")
            {
                use_gpu = true;
            }
            else if (comboBox1.SelectedItem.ToString() == "CPU")
            {
                use_gpu = false;
            }
            comboBox1_last_index = comboBox1.SelectedIndex;


            comboBox_clicked = 0;
        }

        int last_gpu_id = 0;
        int gpu_id_done = 0;
        private void textBox1_TextChanged(object sender, EventArgs e)
        {
            if (is_infer == 1 && gpu_id_done == 0)
            {
                MessageBox.Show("正在推理中，请推理完成后再选择指定GPU重新初始化!", "提示", MessageBoxButtons.OK, MessageBoxIcon.Error);
                gpu_id_done = 1;
                gpu_id = last_gpu_id;
                textBox1.Text = $"{gpu_id}";
                return;
            }

            if (has_model_init == 1 && gpu_id_done == 0)
            {
                MessageBox.Show("模型已初始化，请销毁模型后再进行GPU指定!\n(GPU:x)", "提示", MessageBoxButtons.OK, MessageBoxIcon.Error);
                gpu_id_done = 1;
                gpu_id = last_gpu_id;
                textBox1.Text = $"{gpu_id}";
                return;
            }

            if (gpu_id_done == 0)
            {
                string gpu_id_str = textBox1.Text.ToString();
                if (gpu_id_str.Length != 0)
                {
                    last_gpu_id = gpu_id;
                    try
                    {
                        gpu_id = Int32.Parse(gpu_id_str);
                    }
                    catch (Exception ex)
                    {
                        gpu_id = last_gpu_id;
                        textBox1.Text = $"{gpu_id}";
                        MessageBox.Show("GPU_ID只能输入数字!");
                    }
                }
            }

            gpu_id_done = 0;
        }


        int comboBox2_last_index = 0;

        // Choice of type of model to perform reasoning
        private void comboBox2_SelectedIndexChanged(object sender, EventArgs e)
        {

            if (is_infer == 1 && comboBox_clicked == 0)
            {
                MessageBox.Show("正在推理中，请推理完成后再选择模型类型重新初始化!", "提示", MessageBoxButtons.OK, MessageBoxIcon.Error);
                comboBox_clicked = 1;
                comboBox2.SelectedIndex = comboBox2_last_index;
                return;
            }

            if (has_model_init == 1 && comboBox_clicked == 0)
            {
                MessageBox.Show("模型已初始化，请销毁模型后再进行模型类型选择!\n(det,seg,clas)", "提示", MessageBoxButtons.OK, MessageBoxIcon.Error);
                comboBox_clicked = 1;
                comboBox2.SelectedIndex = comboBox2_last_index;
                return;
            }

            // There are three types
            if (comboBox2.SelectedItem.ToString() == "det")  // Load detection model
            {
                model_type = comboBox2.SelectedItem.ToString();
                paddlex_doing = false;  // Enter non-PADDLEX mode -- detection
            }
            else if (comboBox2.SelectedItem.ToString() == "seg") // Load segmentation model
            {
                model_type = comboBox2.SelectedItem.ToString();
                paddlex_doing = false;
            }
            else if (comboBox2.SelectedItem.ToString() == "clas") // Load recognition model
            {
                model_type = comboBox2.SelectedItem.ToString();
                paddlex_doing = false;
            }
            else if (comboBox2.SelectedItem.ToString() == "mask") // Loading instance split Maskrcnn model
            {
                model_type = comboBox2.SelectedItem.ToString();
                paddlex_doing = false;
            }
            else if (comboBox2.SelectedItem.ToString() == "paddlex") // Load paddlex model
            {
                model_type = comboBox2.SelectedItem.ToString();
                // Repeat Paddlex, do not modify the status
            }
            comboBox2_last_index = comboBox2.SelectedIndex;

            
            comboBox_clicked = 0;
        }

        
        int comboBox3_last_index = 0;
        // Set the detection threshold for target detection
        private void comboBox3_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (is_infer == 1 && comboBox_clicked == 0)
            {
                MessageBox.Show("正在推理中，检测阈值修改将在本次模型推理完成后生效!", "提示", MessageBoxButtons.OK, MessageBoxIcon.Error);

                comboBox_clicked = 1;
                comboBox3.SelectedIndex = comboBox3_last_index;
                return;
            }

            det_threshold = float.Parse(comboBox3.SelectedItem.ToString());
            comboBox3_last_index = comboBox3.SelectedIndex;


            comboBox_clicked = 0;
        }


        // Continuous reasoning interval length - picture folder reasoning
        private void numericUpDown1_ValueChanged(object sender, EventArgs e)
        {
            continue_infer_delay = ((int)numericUpDown1.Value);
        }


        /**********************************************************************/
        /*****************          5.Select control components implementation          ***************/
        /**********************************************************************/
        // Load the model related file folder
        private void button1_Click(object sender, EventArgs e)
        {
            if (is_infer == 1)
            {
                MessageBox.Show("正在推理中，请推理完成后再初始化加载模型!", "提示", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            if (!CheckMaskRCNN_workOnGpu(model_type, use_gpu)) // The Maskrcnn environment is not in the GPU, then an error reminder
            {
                MessageBox.Show("MaskRCNN推理仅支持GPU环境，请重新选择启动环境!\n(因为CPU环境可能存在内存不足，导致推理失败。)", "提示", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            int dir_load_flag = 0;
            string dir_path = null;
            folderBrowserDialog1.Description = "请选择模型文件夹";
            DialogResult folder = folderBrowserDialog1.ShowDialog();
            if (folder == DialogResult.OK || folder == DialogResult.Yes)
            {
                dir_path = folderBrowserDialog1.SelectedPath;
                if (string.IsNullOrEmpty(dir_path))
                {
                    MessageBox.Show("请选择模型路径/模型路径为空!", "提示", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    dir_load_flag = 0;
                }
                else
                {
                    dir_load_flag = 1;
                }
            }

            if (dir_load_flag == 1)
            {
                List<FileInfo> model_lst = new List<FileInfo>();
                List<FileInfo> params_lst = new List<FileInfo>();
                List<FileInfo> cfg_lst = new List<FileInfo>();
                model_lst = getFile(dir_path, ".pdmodel", model_lst);
                params_lst = getFile(dir_path, ".pdiparams", params_lst);
                cfg_lst = getFile(dir_path, ".yml", cfg_lst);
                if (cfg_lst.Count == 0)
                    cfg_lst = getFile(dir_path, ".yaml", cfg_lst);

                if (model_lst.Count != 1 || params_lst.Count != 1)
                {
                    MessageBox.Show("模型文件加载失败!\n请注意模型文件夹下应包含以下文件各一个:\n*.pdmodel, *.pdiparams", "提示");
                    return;
                }

                model_filename = model_lst[0].FullName;
                params_filename = params_lst[0].FullName;

                if (cfg_lst.Count == 0)
                {
                    MessageBox.Show("模型文件加载失败!\n请注意模型文件夹下应包含以下文件:\nmodel.yml/model.yaml", "提示");
                    return;
                }
                else if (cfg_lst.Count > 2) // Excessive YML， > 2 
                {
                    MessageBox.Show("模型文件加载失败!\n请注意模型文件夹下应至多包含以下文件(yml文件个数不得超过2):\nmodel.yml/model.yaml，pipeline.yml/pipeline.yaml", "提示");
                    return;
                }
                else if (cfg_lst.Count == 2) // Processing for the case of multiple YML files
                {
                    for (int i = 0; i < 2; i++)
                    {
                        if (cfg_lst[i].Name == "model.yml" || cfg_lst[i].Name == "model.yaml")
                        {
                            cfg_file = cfg_lst[i].FullName;
                            break;
                        }
                    }
                }
                else if (cfg_lst.Count == 1)
                {
                    cfg_file = cfg_lst[0].FullName;
                }

                int raise_ex_flag = 0;
                int is_Mask = 0; 
                if (has_model_init == 1)
                {
                    DestructModel();

                    try
                    {
                        if (paddlex_doing == true) model_type = "paddlex";
                        if (model_type == "mask")
                        {
                            model_type = "paddlex";
                            is_Mask = 1;
                        }
                        InitModel(model_type, model_filename, params_filename, cfg_file, use_gpu, gpu_id, ref paddlex_model_type[0]);

                        if (is_Mask == 1)
                        {
                            model_type = "mask";
                            is_Mask = 0;
                        }
                        if (model_type == "paddlex")
                        {
                            paddlex_doing = true;
                            model_type = System.Text.Encoding.UTF8.GetString(paddlex_model_type).Split('\0')[0];
                        }
                    }
                    catch (Exception ex)
                    {
                        raise_ex_flag = 1;
                        MessageBox.Show("1.请确定文件中包含有效的模型文件(*.pdmodel, *.pdiparams, *.yml)!\n2.请检查模型文件与模型类型是否一致!\n3.其它原因:GPU号有误，yml中预处理有误...", "模型初始化失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    }
                }
                else
                {

                    try
                    {

                        if (paddlex_doing == true) model_type = "paddlex";
                        if (model_type == "mask")
                        {
                            model_type = "paddlex";
                            is_Mask = 1;
                        }
                        InitModel(model_type, model_filename, params_filename, cfg_file, use_gpu, gpu_id, ref paddlex_model_type[0]);

                        if (is_Mask == 1)
                        {
                            model_type = "mask";
                            is_Mask = 0;
                        }
                        if (model_type == "paddlex")
                        {
                            paddlex_doing = true;

                            model_type = System.Text.Encoding.UTF8.GetString(paddlex_model_type).Split('\0')[0];
                        }
                    }
                    catch (Exception ex)
                    {
                        raise_ex_flag = 1;
                        MessageBox.Show("1.请确定文件中包含有效的模型文件(*.pdmodel, *.pdiparams, *.yml)!\n2.请检查模型文件与模型类型是否一致!\n3.其它原因:GPU号有误，yml中预处理有误...", "模型初始化失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    }
                }

                if (raise_ex_flag == 0)
                {
                    has_model_init = 1;
                    if (use_gpu)
                    {
                        if (is_Mask == 1)
                        {
                            model_type = "mask";
                            is_Mask = 0;
                        }
                        MessageBox.Show($"模型文件已加载到GPU:{gpu_id}!\n(模型类型为: {model_type.Split('\0')[0]})", "提示");
                    }
                    else
                    {
                        MessageBox.Show($"模型类型为: {model_type.Split('\0')[0]}", "提示");
                    }
                    button1.Text = "模型已初始化";
                }
            }
        }

        // Load a single image
        private void button2_Click(object sender, EventArgs e)
        {
            
            if (is_infer == 1)
            {
                MessageBox.Show("正在推理中，请推理完成后再选择推理数据!", "提示", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            int has_load_img_file_flag = 1;
            openFileDialog1.Filter = "(*.png;*.jpg;*.JPEG;*.jpeg)|*.*"; 
            DialogResult dr = openFileDialog1.ShowDialog();
            
            string filename = openFileDialog1.FileName;
            if (dr != System.Windows.Forms.DialogResult.OK || string.IsNullOrEmpty(filename))
            {
                has_load_img_file_flag = 0;
            }

            if (has_load_img_file_flag==1)
            {

                string[] final_tag = filename.Split('.');
                int flag = 0;
                foreach (string type in img_type)
                {
                    if (final_tag[1] == type)
                    {
                        flag = 1;

                        imgfile = filename;

                        pictureBox1.Image = Image.FromFile(imgfile);
                        pictureBox1.SizeMode = PictureBoxSizeMode.Zoom;

                        videofile = null;
                        imgfiles.Clear();

                        MessageBox.Show("图片加载完成!", "提示");

                        button2.Text = "图片已加载"; 
                        button3.Text = "加载图片文件夹"; 
                        button4.Text = "加载视频流"; 
                    }
                }
            }
        }
        
        // Load picture folder
        private void button3_Click(object sender, EventArgs e)
        {
            if (is_infer == 1)
            {
                MessageBox.Show("正在推理中，请推理完成后再选择推理数据!", "提示", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            int img_dir_load_flag = 0;
            string img_dir_path = null;
            folderBrowserDialog1.Description = "请选择模型文件夹";
            DialogResult folder = folderBrowserDialog1.ShowDialog();
            if (folder == DialogResult.OK || folder == DialogResult.Yes)
            {
                img_dir_path = folderBrowserDialog1.SelectedPath;
                if (string.IsNullOrEmpty(img_dir_path)) // 判断是否选择了模型文件
                {
                    MessageBox.Show("请选择图片文件夹路径/图片文件夹路径为空!", "提示", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    img_dir_load_flag = 0;
                }
                else
                {
                    img_dir_load_flag = 1;
                }
            }

            if (img_dir_load_flag == 1)
            {
                List<FileInfo> lst = new List<FileInfo>();
                lst = getFile(img_dir_path, ".jpg", lst);
                lst = getFile(img_dir_path, ".png", lst);
                lst = getFile(img_dir_path, ".JPEG", lst);

                foreach (FileInfo Image_File in lst)
                {
                    imgfiles.Add(Image_File.FullName);
                }

                if (imgfiles.Count == 0)
                {
                    MessageBox.Show("请输入选择非空/包含正确图片类型的图片文件夹!\n(*.png, *.jpg, *.JPEG)", "图片解析失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }

                pictureBox1.Image = Image.FromFile(imgfiles[0]);
                pictureBox1.SizeMode = PictureBoxSizeMode.Zoom;

                imgfile = null;
                videofile = null;
                MessageBox.Show("图片文件夹加载完成!", "提示");

                button3.Text = "图片文件夹已加载";
                button2.Text = "加载图片"; 
                button4.Text = "加载视频流"; 
            }
       
        }

        // Load the video stream
        private void button4_Click(object sender, EventArgs e)
        {
            if (is_infer == 1)
            {
                MessageBox.Show("正在推理中，请推理完成后再选择推理数据!", "提示", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            int has_load_mp4_file_flag = 1;
            openFileDialog1.Filter = "(*.mp4)|*.*";
            DialogResult dr = openFileDialog1.ShowDialog();

            string filename = openFileDialog1.FileName;
            if (dr != System.Windows.Forms.DialogResult.OK || string.IsNullOrEmpty(filename))
            {
                if (string.IsNullOrEmpty(filename)) MessageBox.Show("请选择视频(*.mp4)文件!", "视频路径为空", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                has_load_mp4_file_flag = 0;
            }
            
            if (has_load_mp4_file_flag==1)
            {
                string[] final_tag = filename.Split('.');

                if (final_tag[1] == "mp4")
                {
                    videofile = filename;

                    Bitmap image = null;
                    Mat frame = new Mat();
                    VideoCapture capture = new VideoCapture();
                    capture.Open(videofile);
                    bool read_success = capture.Read(frame); 
                    if (!read_success)
                    {
                        MessageBox.Show("无法读取视频的帧！！！", "提示");
                    }
                    else
                    {
                        image = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(frame);

                        pictureBox1.Image = image;
                        pictureBox1.SizeMode = PictureBoxSizeMode.Zoom;

                        capture = null; 
                        frame = null; 
                        image = null;

                        imgfile = null;
                        imgfiles.Clear();
                        MessageBox.Show("视频加载完成!", "提示");

                        button4.Text = "视频流已加载"; 
                        button2.Text = "加载图片";  
                        button3.Text = "加载图片文件夹"; 
                    }
                }
                else
                {
                    MessageBox.Show("请选择mp4视频文件!", "视频资源加载失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }

            }
        }

        // Perform reasoning
        private void button5_Click(object sender, EventArgs e)
        {
            if (imgfile != null && is_infer==0 && has_model_init == 1)  // Single picture prediction
            {
                Thread infer_one_img_thread = null;
                if (model_type == "det") infer_one_img_thread = new Thread(new ThreadStart(delegate { det_infer_one_img();  }));
                else if (model_type == "seg") infer_one_img_thread = new Thread(new ThreadStart(delegate { seg_infer_one_img(); }));
                else if (model_type == "clas") infer_one_img_thread = new Thread(new ThreadStart(delegate { cls_infer_one_img(); }));
                else if (model_type == "mask") infer_one_img_thread = new Thread(new ThreadStart(delegate { mask_infer_one_img(); }));
                MessageBox.Show("开始图片推理任务!", "提示");
                infer_one_img_thread.Start();
                infer_one_img_flag = 1;

            }
            else if (imgfiles.Count != 0 && is_infer == 0 && has_model_init == 1) // Picture folder prediction
            {
                Thread infer_many_img_thread = null;
                if (model_type == "det") infer_many_img_thread = new Thread(new ThreadStart(delegate { det_infer_many_img(); }));
                else if (model_type == "seg") infer_many_img_thread = new Thread(new ThreadStart(delegate { seg_infer_many_img(); }));
                else if (model_type == "clas") infer_many_img_thread = new Thread(new ThreadStart(delegate { cls_infer_many_img(); }));
                else if (model_type == "mask") infer_many_img_thread = new Thread(new ThreadStart(delegate { mask_infer_many_img(); }));
                MessageBox.Show("开始图片文件夹推理任务!", "提示");
                infer_many_img_thread.Start();
                infer_many_img_flag = 1;
            }
            else if (videofile != null && is_infer == 0 && has_model_init == 1) // Video reasoning
            {
                Thread infer_video_img_thread = null;
                if (model_type == "det") infer_video_img_thread = new Thread(new ThreadStart(delegate { det_infer_video_img(); }));
                else if (model_type == "seg") infer_video_img_thread = new Thread(new ThreadStart(delegate { seg_infer_video_img(); }));
                else if (model_type == "clas") infer_video_img_thread = new Thread(new ThreadStart(delegate { cls_infer_video_img(); }));
                else if (model_type == "mask") infer_video_img_thread = new Thread(new ThreadStart(delegate { mask_infer_video_img(); }));
                MessageBox.Show("开始视频推理任务!", "提示");
                infer_video_img_thread.Start();
                infer_video_img_flag = 1;
            }
            else if (is_infer == 1 && has_model_init == 1)
            {
                if (infer_one_img_flag == 1) MessageBox.Show("正在进行推理任务!", "请勿再执行图片推理任务", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                if (infer_many_img_flag == 1) MessageBox.Show("正在进行推理任务!", "请勿再执行图片文件夹推理任务", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                if (infer_video_img_flag == 1) MessageBox.Show("正在进行推理任务!", "请勿再执行视频推理任务", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
            
            if (has_model_init == 0 && (imgfile == null && imgfiles.Count == 0 && videofile == null))
            {
                MessageBox.Show("请先初始化模型，并选择加载的推理数据后，再点击模型推理!", "推理执行失败", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
            else if (has_model_init == 0 && (imgfile != null || imgfiles.Count != 0 || videofile != null))
            {
                MessageBox.Show("请初始化模型，再点击模型推理!", "推理执行失败", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
            else if (has_model_init != 0 && (imgfile == null && imgfiles.Count == 0 && videofile == null))
            {
                MessageBox.Show("请选择加载的推理数据，再点击模型推理!", "推理执行失败", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
        }

        // Termination of the reasoning
        private void button6_Click(object sender, EventArgs e)
        {
            isBreakInfer = 1;
        }

        // Destroy the initialized model
        private void button7_Click(object sender, EventArgs e)
        {
            if (is_infer == 0)
            {
                if (has_model_init == 1)
                {
                    try
                    {
                        DestructModel();
                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show("当前未初始化模型，无需销毁!", "模型销毁失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    }
                    has_model_init = 0;
                }

                button1.Text = "初始化模型";
            }
            else
            {
                MessageBox.Show("请先中断模型推理，再销毁已初始化的模型!", "提示", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }


        /***********************************************************************/
        /*****************          6.Visual reasoning implementation          **************/
        /***********************************************************************/
        private void det_infer_one_img()
        {
            is_infer = 1;

            byte[] color_map = get_color_map_list(256);

            //Bitmap bmp = new Bitmap(imgfile);
            Bitmap bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(Cv2.ImRead(imgfile));
            byte[] inputData = GetBGRValues(bmp, out int stride);

            float[] resultlist = new float[600];
            IntPtr results = FloatToIntptr(resultlist);
            int[] boxesInfo = new int[1]; // 10 boundingbox
            byte[] labellist = new byte[1000]; 

            int raise_ex_flag = 0; 
            try
            {
                TimeSpan infer_start_time = new TimeSpan(DateTime.Now.Ticks);
                // The fourth parameter is the number of channels for inputting images
                Det_ModelPredict(inputData, bmp.Width, bmp.Height, 3, results, boxesInfo, ref labellist[0]);
                TimeSpan infer_end_time = new TimeSpan(DateTime.Now.Ticks);

                string strGet = System.Text.Encoding.Default.GetString(labellist, 0, labellist.Length); 
                string[] predict_Label_List = strGet.Split(' '); 
                using OpenCvSharp.Mat mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(bmp);
                for (int i = 0; i < boxesInfo[0]; i++)
                {
                    int labelindex = Convert.ToInt32(resultlist[i * 6 + 0]);
                    float score = resultlist[i * 6 + 1];
                    float left = resultlist[i * 6 + 2];
                    float top = resultlist[i * 6 + 3];
                    float right = resultlist[i * 6 + 4];
                    float down = resultlist[i * 6 + 5];

                    if (score > det_threshold)
                    {
                        int[] color_ = { (int)(color_map[(labelindex%256)*3]),
                                             (int)(color_map[(labelindex % 256) * 3 + 1]),
                                             (int)(color_map[(labelindex % 256) * 3 + 2]) };

                        var text_size = Cv2.GetTextSize($"{predict_Label_List[i]}-{labelindex}-{score:f2}",
                                         HersheyFonts.HersheySimplex, 1, 2, out int baseline); 
                        int left_down_x = (int)left + 22;
                        int left_down_y = (int)top + text_size.Height;

                        Cv2.Rectangle(mat, new OpenCvSharp.Rect((int)left, (int)top, (int)right, (int)down), new OpenCvSharp.Scalar(color_[0], color_[1], color_[2]), 2, LineTypes.AntiAlias);//LineTypes.AntiAlias:反锯齿效果
                        Cv2.PutText(mat, $"{predict_Label_List[i]}-{labelindex}-: {score:f2}", new OpenCvSharp.Point(left_down_x, left_down_y), HersheyFonts.HersheySimplex, 1, new OpenCvSharp.Scalar(color_[0], color_[1], color_[2]), 2, LineTypes.Link4);
                    }
                }

                bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(mat);
                if (pictureBox2.Image != null) pictureBox2.Image.Dispose();
                pictureBox2.Image = bmp;
                pictureBox2.SizeMode = PictureBoxSizeMode.Zoom;

                TimeSpan start2end_time = infer_end_time.Subtract(infer_start_time).Duration();
                double cost_milliseconds = start2end_time.TotalMilliseconds;
                
                Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };
                label7.Invoke(AsyncUIDelegate, new object[] { $"{cost_milliseconds:f2}" });
            }
            catch (Exception e)
            {
                raise_ex_flag = 1;

                Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };
                label7.Invoke(AsyncUIDelegate, new object[] { "0.00" });
                MessageBox.Show("1.请检查模型文件与模型类型是否一致!\n2.内存溢出，yml预处理有误，图片格式确保为1/3通道...", "模型运行失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

            isBreakInfer = 0;
            is_infer = 0;
            infer_one_img_flag = 0;
            if (raise_ex_flag == 0) MessageBox.Show("图片推理完成!");
        }

        private void det_infer_many_img()
        {
            is_infer = 1;

            byte[] color_map = get_color_map_list(256);

            int raise_ex_flag = 0;
            try
            {
                foreach (string img_file in imgfiles)
                {
                    if (isBreakInfer == 1) break;

                    Bitmap show_image = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(Cv2.ImRead(img_file));

                    Bitmap bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(Cv2.ImRead(img_file));
                    byte[] inputData = GetBGRValues(bmp, out int stride);

                    float[] resultlist = new float[600];
                    IntPtr results = FloatToIntptr(resultlist);
                    int[] boxesInfo = new int[1];
                    byte[] labellist = new byte[1000];   

                    TimeSpan infer_start_time = new TimeSpan(DateTime.Now.Ticks);
                    
                    Det_ModelPredict(inputData, bmp.Width, bmp.Height, 3, results, boxesInfo, ref labellist[0]);
                    TimeSpan infer_end_time = new TimeSpan(DateTime.Now.Ticks);

                    string strGet = System.Text.Encoding.Default.GetString(labellist, 0, labellist.Length);
                    string[] predict_Label_List = strGet.Split(' '); 
                    using OpenCvSharp.Mat mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(bmp);
                    for (int i = 0; i < boxesInfo[0]; i++) 
                    {
                        int labelindex = Convert.ToInt32(resultlist[i * 6 + 0]);
                        float score = resultlist[i * 6 + 1];
                        float left = resultlist[i * 6 + 2];
                        float top = resultlist[i * 6 + 3];
                        float right = resultlist[i * 6 + 4];  // det -- right down
                        float down = resultlist[i * 6 + 5];

                        if (score > det_threshold)
                        {
                            int[] color_ = { (int)(color_map[(labelindex%256)*3]), 
                                             (int)(color_map[(labelindex % 256) * 3 + 1]), 
                                             (int)(color_map[(labelindex % 256) * 3 + 2]) };

                            
                            var text_size = Cv2.GetTextSize($"{predict_Label_List[i]}-{labelindex}-{score:f2}",
                                             HersheyFonts.HersheySimplex, 1, 2, out int baseline);
                            int left_down_x = (int)left + 22;
                            int left_down_y = (int)top + text_size.Height;

                            Cv2.Rectangle(mat, new OpenCvSharp.Rect((int)left, (int)top, (int)right, (int)down), new OpenCvSharp.Scalar(color_[0], color_[1], color_[2]), 2, LineTypes.AntiAlias);//LineTypes.AntiAlias:反锯齿效果
                            Cv2.PutText(mat, $"{predict_Label_List[i]}-{labelindex}-: {score:f2}", new OpenCvSharp.Point(left_down_x, left_down_y), HersheyFonts.HersheySimplex, 1, new OpenCvSharp.Scalar(color_[0], color_[1], color_[2]), 2, LineTypes.Link4);
                        }
                    }
                    bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(mat);

                    if (pictureBox1.Image != null) pictureBox1.Image.Dispose();
                    pictureBox1.Image = show_image; 
                    pictureBox1.SizeMode = PictureBoxSizeMode.Zoom; 
                    
                    if (pictureBox2.Image != null) pictureBox2.Image.Dispose();
                    pictureBox2.Image = bmp;
                    pictureBox2.SizeMode = PictureBoxSizeMode.Zoom;
                   
                    TimeSpan start2end_time = infer_end_time.Subtract(infer_start_time).Duration();
                    double cost_milliseconds = start2end_time.TotalMilliseconds;
                    
                    Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };
                    label7.Invoke(AsyncUIDelegate, new object[] { $"{cost_milliseconds:f2}" });

                    Thread.Sleep(continue_infer_delay);
                }
            }
            catch (Exception e)
            {
                raise_ex_flag = 1;

                Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };
                label7.Invoke(AsyncUIDelegate, new object[] { "0.00" });
                MessageBox.Show("1.请检查模型文件与模型类型是否一致!\n2.内存溢出，yml预处理有误，图片格式确保为1/3通道...", "模型运行失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

            isBreakInfer = 0;

            is_infer = 0; 
            infer_many_img_flag = 0; 
            if (raise_ex_flag == 0) MessageBox.Show("图片文件夹推理完成!"); 
        }

        private void det_infer_video_img()
        {
            is_infer = 1; 

            byte[] color_map = get_color_map_list(256);

            VideoCapture capture = new VideoCapture();
            capture.Open(videofile); 

            using Mat frame = new Mat();

            int raise_ex_flag = 0;  
            try
            {
                while (true)
                {
                    if (isBreakInfer == 1) break;
                    capture.Read(frame);
                    if (frame.Empty()) break;

                    Bitmap image = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(frame); 
                    if (pictureBox1.Image != null) pictureBox1.Image.Dispose();
                    pictureBox1.Image = image; 
                    pictureBox1.SizeMode = PictureBoxSizeMode.Zoom; 

                    image = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(frame);
                    byte[] inputData = GetBGRValues(image, out int stride);

                    float[] resultlist = new float[600];
                    IntPtr results = FloatToIntptr(resultlist);
                    int[] boxesInfo = new int[1];
                    byte[] labellist = new byte[1000];  

                    TimeSpan infer_start_time = new TimeSpan(DateTime.Now.Ticks);
                    
                    Det_ModelPredict(inputData, image.Width, image.Height, 3, results, boxesInfo, ref labellist[0]);
                    TimeSpan infer_end_time = new TimeSpan(DateTime.Now.Ticks);

                    string strGet = System.Text.Encoding.Default.GetString(labellist, 0, labellist.Length);
                    string[] predict_Label_List = strGet.Split(' ');  
                                                                     
                    using OpenCvSharp.Mat mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(image);
                    for (int i = 0; i < boxesInfo[0]; i++)
                    {
                        int labelindex = Convert.ToInt32(resultlist[i * 6 + 0]);
                        float score = resultlist[i * 6 + 1];
                        float left = resultlist[i * 6 + 2];
                        float top = resultlist[i * 6 + 3];
                        float right = resultlist[i * 6 + 4];
                        float down = resultlist[i * 6 + 5];

                        if (score > det_threshold)
                        {
                            int[] color_ = { (int)(color_map[(labelindex%256)*3]),
                                             (int)(color_map[(labelindex % 256) * 3 + 1]),
                                             (int)(color_map[(labelindex % 256) * 3 + 2]) };

                            
                            var text_size = Cv2.GetTextSize($"{predict_Label_List[i]}-{labelindex}-{score:f2}",
                                             HersheyFonts.HersheySimplex, 1, 2, out int baseline);
                            int left_down_x = (int)left + 22;
                            int left_down_y = (int)top + text_size.Height;

                            Cv2.Rectangle(mat, new OpenCvSharp.Rect((int)left, (int)top, (int)right, (int)down), new OpenCvSharp.Scalar(color_[0], color_[1], color_[2]), 2, LineTypes.AntiAlias);//LineTypes.AntiAlias:反锯齿效果
                            Cv2.PutText(mat, $"{predict_Label_List[i]}-{labelindex}-: {score:f2}", new OpenCvSharp.Point(left_down_x, left_down_y), HersheyFonts.HersheySimplex, 1, new OpenCvSharp.Scalar(color_[0], color_[1], color_[2]), 2, LineTypes.Link4);
                        }
                    }

                    image = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(mat);
                    if (pictureBox2.Image != null) pictureBox2.Image.Dispose();
                    pictureBox2.Image = image;
                    pictureBox2.SizeMode = PictureBoxSizeMode.Zoom;

                    TimeSpan start2end_time = infer_end_time.Subtract(infer_start_time).Duration();
                    double cost_milliseconds = start2end_time.TotalMilliseconds;
                    
                    Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };
                    label7.Invoke(AsyncUIDelegate, new object[] { $"{cost_milliseconds:f2}" });
                }
            }
            catch (Exception e)
            {
                raise_ex_flag = 1; 

                Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };
                label7.Invoke(AsyncUIDelegate, new object[] { "0.00" });
                MessageBox.Show("1.请检查模型文件与模型类型是否一致!\n2.内存溢出，yml预处理有误，图片格式确保为1/3通道...", "模型运行失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

            isBreakInfer = 0;

            is_infer = 0;
            infer_video_img_flag = 0; 
            if (raise_ex_flag == 0) MessageBox.Show("视频推理完成!"); 
        }


        // Fixed size display: Short_side: 512
        private void cls_infer_one_img()
        {
            is_infer = 1;

            byte[] color_map = get_color_map_list(256);

            //Bitmap bmp = new Bitmap(imgfile);
            Bitmap bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(Cv2.ImRead(imgfile));

            // resize()
            var short_side = bmp.Width > bmp.Height ? bmp.Height : bmp.Width;
            double resize_scale = 512.0 / short_side;

            OpenCvSharp.Mat input_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(bmp);
            OpenCvSharp.Mat output_mat = new Mat();

            int new_height = (int)(bmp.Height * resize_scale);
            int new_width = (int)(bmp.Width * resize_scale);
            Cv2.Resize(input_mat, output_mat, new OpenCvSharp.Size(new_width, new_height));
            bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(output_mat);

            byte[] inputData = GetBGRValues(bmp, out int stride);

            float[] pre_score = new float[1];
            int[] pre_category_id = new int[1];
            byte[] pre_category = new byte[200]; 

            int raise_ex_flag = 0;
            try
            {
                TimeSpan infer_start_time = new TimeSpan(DateTime.Now.Ticks);

                Cls_ModelPredict(inputData, bmp.Width, bmp.Height, 3, ref pre_score[0], ref pre_category[0], ref pre_category_id[0]);
                TimeSpan infer_end_time = new TimeSpan(DateTime.Now.Ticks);

                string category_strGet = System.Text.Encoding.Default.GetString(pre_category, 0, pre_category.Length).Split('\0')[0];   
                OpenCvSharp.Mat mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(bmp);

                
                int[] color_ = { (int)(color_map[(pre_category_id[0]%256)*3]),
                                        (int)(color_map[(pre_category_id[0] % 256) * 3 + 1]),
                                        (int)(color_map[(pre_category_id[0] % 256) * 3 + 2]) };
                
                var text_size  = Cv2.GetTextSize($"{category_strGet}-{pre_category_id[0]}-{pre_score[0]:f2}",
                                 HersheyFonts.HersheySimplex, 1, 2, out int baseline); 

                int left_down_x = bmp.Width - text_size.Width;
                int left_down_y = text_size.Height;

                Cv2.PutText(mat, $"{category_strGet}-{pre_category_id[0]}-{pre_score[0]:f2}", new OpenCvSharp.Point(left_down_x, left_down_y), HersheyFonts.HersheySimplex, 1, new OpenCvSharp.Scalar(color_[0], color_[1], color_[2]), 2, LineTypes.Link4);
                
                bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(mat);
                if (pictureBox2.Image != null) pictureBox2.Image.Dispose();
                pictureBox2.Image = bmp;
                pictureBox2.SizeMode = PictureBoxSizeMode.Zoom;

                input_mat = null;
                output_mat = null;
                mat = null;
                inputData = null;
                pre_score = null;
                pre_category_id = null;
                pre_category = null;

                TimeSpan start2end_time = infer_end_time.Subtract(infer_start_time).Duration();
                double cost_milliseconds = start2end_time.TotalMilliseconds;

                Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };
                label7.Invoke(AsyncUIDelegate, new object[] { $"{cost_milliseconds:f2}" });

            }
            catch (Exception e)
            {
                raise_ex_flag = 1;

                Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };
                label7.Invoke(AsyncUIDelegate, new object[] { "0.00" });
                MessageBox.Show("1.请检查模型文件与模型类型是否一致!\n2.内存溢出，yml预处理有误，图片格式确保为1/3通道...", "模型运行失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

            isBreakInfer = 0;
            is_infer = 0; 
            infer_one_img_flag = 0;
            if (raise_ex_flag == 0) MessageBox.Show("图片推理完成!"); 
        }


        private void cls_infer_many_img()
        {
            is_infer = 1;

            byte[] color_map = get_color_map_list(256);

            int raise_ex_flag = 0; 
            try
            {
                foreach (string img_file in imgfiles)
                {
                    if (isBreakInfer == 1) break;

                    //Bitmap bmp = new Bitmap(img_file);
                    Bitmap bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(Cv2.ImRead(img_file));

                    // resize()
                    var short_side = bmp.Width > bmp.Height ? bmp.Height : bmp.Width;
                    double resize_scale = 512.0 / short_side;

                    OpenCvSharp.Mat input_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(bmp);
                    OpenCvSharp.Mat output_mat = new Mat();

                    int new_height = (int)(bmp.Height * resize_scale);
                    int new_width = (int)(bmp.Width * resize_scale);
                    Cv2.Resize(input_mat, output_mat, new OpenCvSharp.Size(new_width, new_height));
                    bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(output_mat);

                    byte[] inputData = GetBGRValues(bmp, out int stride);

                    float[] pre_score = new float[1];
                    int[] pre_category_id = new int[1];
                    byte[] pre_category = new byte[200]; 

                    TimeSpan infer_start_time = new TimeSpan(DateTime.Now.Ticks);
 
                    Cls_ModelPredict(inputData, bmp.Width, bmp.Height, 3, ref pre_score[0], ref pre_category[0], ref pre_category_id[0]);
                    TimeSpan infer_end_time = new TimeSpan(DateTime.Now.Ticks);

                    string category_strGet = System.Text.Encoding.Default.GetString(pre_category, 0, pre_category.Length).Split('\0')[0]; 
                    OpenCvSharp.Mat mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(bmp);

                    
                    int[] color_ = { (int)(color_map[(pre_category_id[0]%256)*3]),
                                            (int)(color_map[(pre_category_id[0] % 256) * 3 + 1]),
                                            (int)(color_map[(pre_category_id[0] % 256) * 3 + 2]) };
                    
                    var text_size = Cv2.GetTextSize($"{category_strGet}-{pre_category_id[0]}-:{pre_score[0]:f2}",
                                     HersheyFonts.HersheySimplex, 1, 2, out int baseline);

                    int left_down_x = bmp.Width - text_size.Width;
                    int left_down_y = text_size.Height;

                    Cv2.PutText(mat, $"{category_strGet}-{pre_category_id[0]}-:{pre_score[0]:f2}", new OpenCvSharp.Point(left_down_x, left_down_y), HersheyFonts.HersheySimplex, 1, new OpenCvSharp.Scalar(color_[0], color_[1], color_[2]), 2, LineTypes.Link4);

                    bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(mat);

                    Bitmap show_image = new Bitmap(img_file);
                    if (pictureBox1.Image != null) pictureBox1.Image.Dispose();
                    pictureBox1.Image = show_image; 
                    pictureBox1.SizeMode = PictureBoxSizeMode.Zoom; 

                    if (pictureBox2.Image != null) pictureBox2.Image.Dispose();
                    pictureBox2.Image = bmp;
                    pictureBox2.SizeMode = PictureBoxSizeMode.Zoom;

                    mat = null;
                    inputData = null;
                    pre_score = null;
                    pre_category_id = null;
                    pre_category = null;

                    TimeSpan start2end_time = infer_end_time.Subtract(infer_start_time).Duration();
                    double cost_milliseconds = start2end_time.TotalMilliseconds;
                    
                    Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };
                    label7.Invoke(AsyncUIDelegate, new object[] { $"{cost_milliseconds:f2}" });

                    Thread.Sleep(continue_infer_delay);
                }
            }
            catch (Exception e)
            {
                raise_ex_flag = 1;

                Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };
                label7.Invoke(AsyncUIDelegate, new object[] { "0.00" });
                MessageBox.Show("1.请检查模型文件与模型类型是否一致!\n2.内存溢出，yml预处理有误，图片格式确保为1/3通道...", "模型运行失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

            isBreakInfer = 0; 

            is_infer = 0;
            infer_many_img_flag = 0; 
            if (raise_ex_flag == 0) MessageBox.Show("图片文件夹推理完成!");
        }


        private void cls_infer_video_img()
        {
            is_infer = 1;

            byte[] color_map = get_color_map_list(256);

            VideoCapture capture = new VideoCapture();
            capture.Open(videofile); 

            using Mat frame = new Mat();

            int raise_ex_flag = 0; 
            try
            {
                while (true)
                {
                    if (isBreakInfer == 1) break;
                    capture.Read(frame);
                    if (frame.Empty()) break;

                    Bitmap image = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(frame); 
                    if (pictureBox1.Image != null) pictureBox1.Image.Dispose();
                    pictureBox1.Image = image; 
                    pictureBox1.SizeMode = PictureBoxSizeMode.Zoom; 

                    image = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(frame);

                    // resize()
                    var short_side = image.Width > image.Height ? image.Height : image.Width;
                    double resize_scale = 512.0 / short_side;

                    OpenCvSharp.Mat input_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(image);
                    OpenCvSharp.Mat output_mat = new Mat();

                    int new_height = (int)(image.Height * resize_scale);
                    int new_width = (int)(image.Width * resize_scale);
                    Cv2.Resize(input_mat, output_mat, new OpenCvSharp.Size(new_width, new_height));
                    image = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(output_mat);

                    byte[] inputData = GetBGRValues(image, out int stride);

                    float[] pre_score = new float[1];
                    int[] pre_category_id = new int[1];
                    byte[] pre_category = new byte[200];    

                    TimeSpan infer_start_time = new TimeSpan(DateTime.Now.Ticks);
                    
                    Cls_ModelPredict(inputData, image.Width, image.Height, 3, ref pre_score[0], ref pre_category[0], ref pre_category_id[0]);
                    TimeSpan infer_end_time = new TimeSpan(DateTime.Now.Ticks);

                    string category_strGet = System.Text.Encoding.Default.GetString(pre_category, 0, pre_category.Length).Split('\0')[0];  
                    OpenCvSharp.Mat mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(image);

                    int[] color_ = { (int)(color_map[(pre_category_id[0]%256)*3]),
                                        (int)(color_map[(pre_category_id[0] % 256) * 3 + 1]),
                                        (int)(color_map[(pre_category_id[0] % 256) * 3 + 2]) };
                    
                    var text_size = Cv2.GetTextSize($"{category_strGet}-{pre_category_id[0]}-:{pre_score[0]:f2}",
                                     HersheyFonts.HersheySimplex, 1, 2, out int baseline);

                    int left_down_x = image.Width - text_size.Width;
                    int left_down_y = text_size.Height;

                    Cv2.PutText(mat, $"{category_strGet}-{pre_category_id[0]}-:{pre_score[0]:f2}", new OpenCvSharp.Point(left_down_x, left_down_y), HersheyFonts.HersheySimplex, 1, new OpenCvSharp.Scalar(color_[0], color_[1], color_[2]), 2, LineTypes.Link4);

                    image = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(mat);

                    pictureBox2.Image = image;
                    pictureBox2.SizeMode = PictureBoxSizeMode.Zoom;

                    mat = null;
                    inputData = null;
                    pre_score = null;
                    pre_category_id = null;
                    pre_category = null;

                    TimeSpan start2end_time = infer_end_time.Subtract(infer_start_time).Duration();
                    double cost_milliseconds = start2end_time.TotalMilliseconds;
                    
                    Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };
                    label7.Invoke(AsyncUIDelegate, new object[] { $"{cost_milliseconds:f2}" });
                }
            }
            catch (Exception e)
            {
                raise_ex_flag = 1; 

                Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };
                label7.Invoke(AsyncUIDelegate, new object[] { "0.00" });
                MessageBox.Show("1.请检查模型文件与模型类型是否一致!\n2.内存溢出，yml预处理有误，图片格式确保为1/3通道...", "模型运行失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

            isBreakInfer = 0;

            // DestructModel();

            is_infer = 0;
            infer_video_img_flag = 0;
            if (raise_ex_flag == 0) MessageBox.Show("视频推理完成!");
        }


        // Fixed size display: Short_side: 512
        private void seg_infer_one_img()
        {
            is_infer = 1;

            byte[] color_map = get_color_map_list(256);

            
            //Bitmap origin_bmp = new Bitmap(imgfile);
            Bitmap origin_bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(Cv2.ImRead(imgfile));
            Bitmap input_bmp = null;

            // resize()
            OpenCvSharp.Mat input_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(origin_bmp);
            OpenCvSharp.Mat output_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(origin_bmp);

            Cv2.Resize(input_mat, output_mat, new OpenCvSharp.Size(512, 512));
            input_bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(output_mat);
            input_mat = null;
            output_mat = null;

            byte[] inputData = GetBGRValues(input_bmp, out int stride);

            byte[] output_map = new byte[input_bmp.Height * input_bmp.Width];   

            int raise_ex_flag = 0; 
            try
            {
                TimeSpan infer_start_time = new TimeSpan(DateTime.Now.Ticks);
                
                Seg_ModelPredict(inputData, input_bmp.Width, input_bmp.Height, 3, ref output_map[0]);
                TimeSpan infer_end_time = new TimeSpan(DateTime.Now.Ticks);

                
                input_bmp = CreateBitmap(output_map, input_bmp.Width, input_bmp.Height, color_map); 

                input_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(input_bmp);
                output_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(input_bmp); 
                Cv2.Resize(input_mat, output_mat, new OpenCvSharp.Size(origin_bmp.Width, origin_bmp.Height)); 
                input_mat = null;  

                input_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(origin_bmp); 
                //OpenCvSharp.Mat add_mat = new Mat(); 

                Cv2.AddWeighted(output_mat, 1.0, input_mat, 0.35, 1, output_mat);  
                input_bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(output_mat);

                if (pictureBox2.Image != null) pictureBox2.Image.Dispose();
                pictureBox2.Image = input_bmp;
                pictureBox2.SizeMode = PictureBoxSizeMode.Zoom;

                
                TimeSpan start2end_time = infer_end_time.Subtract(infer_start_time).Duration();
                double cost_milliseconds = start2end_time.TotalMilliseconds;

                Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };
                label7.Invoke(AsyncUIDelegate, new object[] { $"{cost_milliseconds:f2}" });
            }
            catch (Exception e)
            {
                raise_ex_flag = 1;

                Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };
                label7.Invoke(AsyncUIDelegate, new object[] { "0.00" });
                MessageBox.Show("1.请检查模型文件与模型类型是否一致!\n2.内存溢出，yml预处理有误，图片格式确保为1/3通道...", "模型运行失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

            isBreakInfer = 0; 
            is_infer = 0;
            infer_one_img_flag = 0;
            if (raise_ex_flag == 0) MessageBox.Show("图片推理完成!");
        }


        private void seg_infer_many_img()
        {
            is_infer = 1; 

            byte[] color_map = get_color_map_list(256);

            int raise_ex_flag = 0;  
            try
            {
                foreach (string img_file in imgfiles)
                {
                    if (isBreakInfer == 1) break; 

                    Bitmap origin_bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(Cv2.ImRead(img_file));
                    Bitmap input_bmp = null;

                    // resize()
                    OpenCvSharp.Mat input_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(origin_bmp);
                    OpenCvSharp.Mat output_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(origin_bmp);

                    Cv2.Resize(input_mat, output_mat, new OpenCvSharp.Size(512, 512));
                    input_bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(output_mat);
                    input_mat = null;
                    output_mat = null;

                    byte[] inputData = GetBGRValues(input_bmp, out int stride);

                    byte[] output_map = new byte[input_bmp.Height * input_bmp.Width];    

                    TimeSpan infer_start_time = new TimeSpan(DateTime.Now.Ticks);
                    
                    Seg_ModelPredict(inputData, input_bmp.Width, input_bmp.Height, 3, ref output_map[0]);
                    TimeSpan infer_end_time = new TimeSpan(DateTime.Now.Ticks);

                    
                    input_bmp = CreateBitmap(output_map, input_bmp.Width, input_bmp.Height, color_map); 

                    input_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(input_bmp);
                    output_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(input_bmp); 
                    Cv2.Resize(input_mat, output_mat, new OpenCvSharp.Size(origin_bmp.Width, origin_bmp.Height)); 
                    input_mat = null;  

                    input_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(origin_bmp); 

                    Cv2.AddWeighted(output_mat, 1.0, input_mat, 0.35, 1, output_mat);  
                    input_bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(output_mat);

                    
                    if (pictureBox1.Image != null) pictureBox1.Image.Dispose();
                    pictureBox1.Image = origin_bmp; 
                    pictureBox1.SizeMode = PictureBoxSizeMode.Zoom; 

                    if (pictureBox2.Image != null) pictureBox2.Image.Dispose();
                    pictureBox2.Image = input_bmp;
                    pictureBox2.SizeMode = PictureBoxSizeMode.Zoom;

                    TimeSpan start2end_time = infer_end_time.Subtract(infer_start_time).Duration();
                    double cost_milliseconds = start2end_time.TotalMilliseconds;

                    Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };
                    label7.Invoke(AsyncUIDelegate, new object[] { $"{cost_milliseconds:f2}" });

                    Thread.Sleep(continue_infer_delay);
                }
            }
            catch (Exception e)
            {
                raise_ex_flag = 1;

                Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };
                label7.Invoke(AsyncUIDelegate, new object[] { "0.00" });
                MessageBox.Show("1.请检查模型文件与模型类型是否一致!\n2.内存溢出，yml预处理有误，图片格式确保为1/3通道...", "模型运行失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

            isBreakInfer = 0; 

            is_infer = 0; 
            infer_many_img_flag = 0;  
            if (raise_ex_flag == 0) MessageBox.Show("图片文件夹推理完成!"); 
        }

        private void seg_infer_video_img()
        {
            is_infer = 1; 

            byte[] color_map = get_color_map_list(256);

            VideoCapture capture = new VideoCapture();
            capture.Open(videofile); 

            using Mat frame = new Mat();

            int raise_ex_flag = 0;  
            try
            {
                while (true)
                {
                    if (isBreakInfer == 1) break;
                    capture.Read(frame);
                    if (frame.Empty()) break;

                    Bitmap origin_bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(frame);

                    Bitmap input_bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(frame);
                    // resize()
                    OpenCvSharp.Mat input_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(origin_bmp);
                    OpenCvSharp.Mat output_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(origin_bmp);

                    Cv2.Resize(input_mat, output_mat, new OpenCvSharp.Size(512, 512));
                    input_bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(output_mat);
                    input_mat = null;
                    output_mat = null;

                    byte[] inputData = GetBGRValues(input_bmp, out int stride);

                    byte[] output_map = new byte[input_bmp.Height * input_bmp.Width];  

                    TimeSpan infer_start_time = new TimeSpan(DateTime.Now.Ticks);
                    
                    Seg_ModelPredict(inputData, input_bmp.Width, input_bmp.Height, 3, ref output_map[0]);
                    TimeSpan infer_end_time = new TimeSpan(DateTime.Now.Ticks);

                    
                    input_bmp = CreateBitmap(output_map, input_bmp.Width, input_bmp.Height, color_map); 

                    input_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(input_bmp);
                    output_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(input_bmp); 
                    Cv2.Resize(input_mat, output_mat, new OpenCvSharp.Size(origin_bmp.Width, origin_bmp.Height)); 
                    input_mat = null;  

                    input_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(origin_bmp); 

                    Cv2.AddWeighted(output_mat, 1.0, input_mat, 0.35, 1, output_mat); 
                    input_bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(output_mat);
                    

                    if (pictureBox1.Image != null) pictureBox1.Image.Dispose();
                    pictureBox1.Image = origin_bmp; 
                    pictureBox1.SizeMode = PictureBoxSizeMode.Zoom; 

                    if (pictureBox2.Image != null) pictureBox2.Image.Dispose();
                    pictureBox2.Image = input_bmp;
                    pictureBox2.SizeMode = PictureBoxSizeMode.Zoom;

                    
                    TimeSpan start2end_time = infer_end_time.Subtract(infer_start_time).Duration();
                    double cost_milliseconds = start2end_time.TotalMilliseconds;

                    Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };
                    label7.Invoke(AsyncUIDelegate, new object[] { $"{cost_milliseconds:f2}" });
                }
            }
            catch (Exception e)
            {
                raise_ex_flag = 1;

                Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };
                label7.Invoke(AsyncUIDelegate, new object[] { "0.00" });
                MessageBox.Show("1.请检查模型文件与模型类型是否一致!\n2.内存溢出，yml预处理有误，图片格式确保为1/3通道...", "模型运行失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

            isBreakInfer = 0;

            // DestructModel();

            is_infer = 0;
            infer_video_img_flag = 0;
            if (raise_ex_flag == 0) MessageBox.Show("视频推理完成!");
        }


        // Maskrcnn detection single picture - GPU reasoning is normal
        private void mask_infer_one_img()
        {
            is_infer = 1; 

            byte[] color_map = get_color_map_list(256);

            Bitmap origin_bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(Cv2.ImRead(imgfile));
            Bitmap input_bmp = null;

            // resize()
            OpenCvSharp.Mat input_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(origin_bmp);
            OpenCvSharp.Mat output_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(origin_bmp);

            input_bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(output_mat);
            byte[] inputData = GetBGRValues(origin_bmp, out int stride);

            float[] resultlist = new float[600];
            IntPtr results = FloatToIntptr(resultlist);

            byte[] mask_results = new byte[input_bmp.Height * input_bmp.Width];  

            int[] boxesInfo = new int[1];
            byte[] labellist = new byte[1000]; 

            int raise_ex_flag = 0;
            try
            {
                TimeSpan infer_start_time = new TimeSpan(DateTime.Now.Ticks);
                Mask_ModelPredict(inputData, input_bmp.Width, input_bmp.Height, 3, results, ref mask_results[0], boxesInfo, ref labellist[0]);
                TimeSpan infer_end_time = new TimeSpan(DateTime.Now.Ticks);

                input_bmp = CreateBitmap(mask_results, input_bmp.Width, input_bmp.Height, color_map);

                output_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(input_bmp); 
                input_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(origin_bmp); 

                Cv2.AddWeighted(output_mat, 0.65, input_mat, 0.35, 1, output_mat); 
                input_bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(output_mat);

                string strGet = System.Text.Encoding.Default.GetString(labellist, 0, labellist.Length); 
                string[] predict_Label_List = strGet.Split(' ');

                using OpenCvSharp.Mat mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(input_bmp);
                for (int i = 0; i < boxesInfo[0]; i++)
                {
                    int labelindex = Convert.ToInt32(resultlist[i * 6 + 0]);
                    float score = resultlist[i * 6 + 1];
                    float left = resultlist[i * 6 + 2];
                    float top = resultlist[i * 6 + 3];
                    float right = resultlist[i * 6 + 4];
                    float down = resultlist[i * 6 + 5];

                    if (score > det_threshold)
                    {
                        labelindex += 1; // Mask rcnn contains background, so add 1
                        int[] color_ = { (int)(color_map[(labelindex%256)*3]),
                                             (int)(color_map[(labelindex % 256) * 3 + 1]),
                                             (int)(color_map[(labelindex % 256) * 3 + 2]) };

                        labelindex -= 1; 
                        
                        var text_size = Cv2.GetTextSize($"{predict_Label_List[i]}-{labelindex}-{score:f2}",
                                         HersheyFonts.HersheySimplex, 1, 2, out int baseline);
                                         
                        int left_down_x = (int)left + 22;
                        int left_down_y = (int)top + text_size.Height;

                        Cv2.Rectangle(mat, new OpenCvSharp.Rect((int)left, (int)top, (int)right, (int)down), new OpenCvSharp.Scalar(color_[0], color_[1], color_[2]), 2, LineTypes.AntiAlias);//LineTypes.AntiAlias:反锯齿效果
                        Cv2.PutText(mat, $"{predict_Label_List[i]}-{labelindex}-: {score:f2}", new OpenCvSharp.Point(left_down_x, left_down_y), HersheyFonts.HersheySimplex, 1, new OpenCvSharp.Scalar(color_[0], color_[1], color_[2]), 2, LineTypes.Link4);
                    }
                }

                input_bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(mat);
                if (pictureBox2.Image != null) pictureBox2.Image.Dispose();
                pictureBox2.Image = input_bmp;
                pictureBox2.SizeMode = PictureBoxSizeMode.Zoom;

                TimeSpan start2end_time = infer_end_time.Subtract(infer_start_time).Duration();
                double cost_milliseconds = start2end_time.TotalMilliseconds;

                Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };
                label7.Invoke(AsyncUIDelegate, new object[] { $"{cost_milliseconds:f2}" });

            }
            catch (Exception e)
            {
                raise_ex_flag = 1;

                Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };
                label7.Invoke(AsyncUIDelegate, new object[] { "0.00" });
                MessageBox.Show("1.请检查模型文件与模型类型是否一致!\n2.内存溢出，yml预处理有误，图片格式确保为1/3通道...", "模型运行失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

            isBreakInfer = 0; 
            is_infer = 0; 
            infer_one_img_flag = 0; 
            if (raise_ex_flag == 0) MessageBox.Show("图片推理完成!");
        }


        private void mask_infer_many_img()
        {
            is_infer = 1; 

            byte[] color_map = get_color_map_list(256);

            int raise_ex_flag = 0;  
            try
            {
                foreach (string img_file in imgfiles)
                {
                    if (isBreakInfer == 1) break;  

                    Bitmap origin_bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(Cv2.ImRead(img_file));
                    Bitmap input_bmp = null;

                    // resize()
                    OpenCvSharp.Mat input_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(origin_bmp);
                    OpenCvSharp.Mat output_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(origin_bmp);

                    input_bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(output_mat);
                    byte[] inputData = GetBGRValues(origin_bmp, out int stride);

                    float[] resultlist = new float[600];
                    IntPtr results = FloatToIntptr(resultlist);

                    byte[] mask_results = new byte[input_bmp.Height * input_bmp.Width];    

                    int[] boxesInfo = new int[1]; 
                    byte[] labellist = new byte[1000];  

                    TimeSpan infer_start_time = new TimeSpan(DateTime.Now.Ticks);

                    Mask_ModelPredict(inputData, input_bmp.Width, input_bmp.Height, 3, results, ref mask_results[0], boxesInfo, ref labellist[0]);
                    TimeSpan infer_end_time = new TimeSpan(DateTime.Now.Ticks);

                    input_bmp = CreateBitmap(mask_results, input_bmp.Width, input_bmp.Height, color_map);

                    output_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(input_bmp); 
                    input_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(origin_bmp); 

                    Cv2.AddWeighted(output_mat, 1.0, input_mat, 0.35, 1, output_mat);  
                    input_bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(output_mat);

                    string strGet = System.Text.Encoding.Default.GetString(labellist, 0, labellist.Length);    
                    string[] predict_Label_List = strGet.Split(' ');  

                    using OpenCvSharp.Mat mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(input_bmp);
                    for (int i = 0; i < boxesInfo[0]; i++) 
                    {
                        int labelindex = Convert.ToInt32(resultlist[i * 6 + 0]);
                        float score = resultlist[i * 6 + 1];
                        float left = resultlist[i * 6 + 2];
                        float top = resultlist[i * 6 + 3];
                        float right = resultlist[i * 6 + 4];
                        float down = resultlist[i * 6 + 5];

                        if (score > det_threshold)
                        {
                            labelindex += 1;
                            int[] color_ = { (int)(color_map[(labelindex%256)*3]),
                                             (int)(color_map[(labelindex % 256) * 3 + 1]),
                                             (int)(color_map[(labelindex % 256) * 3 + 2]) };

                            labelindex -= 1; 

                            var text_size = Cv2.GetTextSize($"{predict_Label_List[i]}-{labelindex}-{score:f2}",
                                             HersheyFonts.HersheySimplex, 1, 2, out int baseline);  
                                             
                            int left_down_x = (int)left + 22; 
                            int left_down_y = (int)top + text_size.Height;

                            Cv2.Rectangle(mat, new OpenCvSharp.Rect((int)left, (int)top, (int)right, (int)down), new OpenCvSharp.Scalar(color_[0], color_[1], color_[2]), 2, LineTypes.AntiAlias);//LineTypes.AntiAlias:反锯齿效果
                            Cv2.PutText(mat, $"{predict_Label_List[i]}-{labelindex}-: {score:f2}", new OpenCvSharp.Point(left_down_x, left_down_y), HersheyFonts.HersheySimplex, 1, new OpenCvSharp.Scalar(color_[0], color_[1], color_[2]), 2, LineTypes.Link4);
                        }
                    }

                    if (pictureBox1.Image != null) pictureBox1.Image.Dispose();
                    pictureBox1.Image = origin_bmp; 
                    pictureBox1.SizeMode = PictureBoxSizeMode.Zoom; 

                    input_bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(mat);
                    if (pictureBox2.Image != null) pictureBox2.Image.Dispose();
                    pictureBox2.Image = input_bmp;
                    pictureBox2.SizeMode = PictureBoxSizeMode.Zoom;

                    TimeSpan start2end_time = infer_end_time.Subtract(infer_start_time).Duration();
                    double cost_milliseconds = start2end_time.TotalMilliseconds;

                    Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };
                    label7.Invoke(AsyncUIDelegate, new object[] { $"{cost_milliseconds:f2}" });

                    Thread.Sleep(continue_infer_delay);
                }
            }
            catch (Exception e)
            {
                raise_ex_flag = 1;

                Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };
                label7.Invoke(AsyncUIDelegate, new object[] { "0.00" });
                MessageBox.Show("1.请检查模型文件与模型类型是否一致!\n2.内存溢出，yml预处理有误，图片格式确保为1/3通道...", "模型运行失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

            isBreakInfer = 0;

            is_infer = 0;
            infer_many_img_flag = 0; 
            if (raise_ex_flag == 0) MessageBox.Show("图片文件夹推理完成!");
        }

        private void mask_infer_video_img()
        {
            is_infer = 1;

            byte[] color_map = get_color_map_list(256);

            VideoCapture capture = new VideoCapture();
            capture.Open(videofile);

            using Mat frame = new Mat();

            int raise_ex_flag = 0;
            try
            {
                while (true)
                {
                    if (isBreakInfer == 1) break;
                    capture.Read(frame);
                    if (frame.Empty()) break;

                    Bitmap origin_bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(frame);
                    Bitmap input_bmp = null;

                    OpenCvSharp.Mat input_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(origin_bmp);
                    OpenCvSharp.Mat output_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(origin_bmp);

                    input_bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(output_mat);
                    byte[] inputData = GetBGRValues(origin_bmp, out int stride);

                    float[] resultlist = new float[600];
                    IntPtr results = FloatToIntptr(resultlist);

                    byte[] mask_results = new byte[input_bmp.Height * input_bmp.Width];

                    int[] boxesInfo = new int[1];
                    byte[] labellist = new byte[1000];

                    TimeSpan infer_start_time = new TimeSpan(DateTime.Now.Ticks);
                    
                    Mask_ModelPredict(inputData, input_bmp.Width, input_bmp.Height, 3, results, ref mask_results[0], boxesInfo, ref labellist[0]);
                    TimeSpan infer_end_time = new TimeSpan(DateTime.Now.Ticks);

                    input_bmp = CreateBitmap(mask_results, input_bmp.Width, input_bmp.Height, color_map);

                    output_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(input_bmp);
                    input_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(origin_bmp); 

                    Cv2.AddWeighted(output_mat, 1.0, input_mat, 0.35, 1, output_mat);
                    input_bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(output_mat);

                    string strGet = System.Text.Encoding.Default.GetString(labellist, 0, labellist.Length);
                    string[] predict_Label_List = strGet.Split(' ');
                    using OpenCvSharp.Mat mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(input_bmp);
                    for (int i = 0; i < boxesInfo[0]; i++)
                    {
                        int labelindex = Convert.ToInt32(resultlist[i * 6 + 0]);
                        float score = resultlist[i * 6 + 1];
                        float left = resultlist[i * 6 + 2];
                        float top = resultlist[i * 6 + 3];
                        float right = resultlist[i * 6 + 4];
                        float down = resultlist[i * 6 + 5];

                        if (score > det_threshold)
                        {
                            labelindex += 1;
                            int[] color_ = { (int)(color_map[(labelindex%256)*3]),
                                             (int)(color_map[(labelindex % 256) * 3 + 1]),
                                             (int)(color_map[(labelindex % 256) * 3 + 2]) };

                            labelindex -= 1;
                            var text_size = Cv2.GetTextSize($"{predict_Label_List[i]}-{labelindex}-{score:f2}",
                                             HersheyFonts.HersheySimplex, 1, 2, out int baseline);
                            int left_down_x = (int)left + 22;
                            int left_down_y = (int)top + text_size.Height;

                            Cv2.Rectangle(mat, new OpenCvSharp.Rect((int)left, (int)top, (int)right, (int)down), new OpenCvSharp.Scalar(color_[0], color_[1], color_[2]), 2, LineTypes.AntiAlias);//LineTypes.AntiAlias:反锯齿效果
                            Cv2.PutText(mat, $"{predict_Label_List[i]}-{labelindex}-: {score:f2}", new OpenCvSharp.Point(left_down_x, left_down_y), HersheyFonts.HersheySimplex, 1, new OpenCvSharp.Scalar(color_[0], color_[1], color_[2]), 2, LineTypes.Link4);
                        }
                    }

                    if (pictureBox1.Image != null) pictureBox1.Image.Dispose();
                    pictureBox1.Image = origin_bmp;
                    pictureBox1.SizeMode = PictureBoxSizeMode.Zoom;

                    input_bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(mat);
                    if (pictureBox2.Image != null) pictureBox2.Image.Dispose();
                    pictureBox2.Image = input_bmp;
                    pictureBox2.SizeMode = PictureBoxSizeMode.Zoom;

                    TimeSpan start2end_time = infer_end_time.Subtract(infer_start_time).Duration();
                    double cost_milliseconds = start2end_time.TotalMilliseconds;
    
                    Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };
                    label7.Invoke(AsyncUIDelegate, new object[] { $"{cost_milliseconds:f2}" });
                }
            }
            catch (Exception e)
            {
                raise_ex_flag = 1;

                Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };
                label7.Invoke(AsyncUIDelegate, new object[] { "0.00" });
                MessageBox.Show("1.Please check if the model file is consistent with the model type!\n2.Memory overflow, YML pretreatment incorrectly, the picture format is ensured as 1/3 channel...", "Model run failure", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

            isBreakInfer = 0;

            is_infer = 0;
            infer_video_img_flag = 0;
            if (raise_ex_flag == 0) MessageBox.Show("视频推理完成!");
        }


        /**********************************************************************/
        /*****************          7.Part of the reasoning component function          ***************/
        /**********************************************************************/
        ///   <summary>
        ///  Read the data from the memory stream
        ///   </summary>
        ///   <param name="curStream"></param>
        ///   <param name="startPosition"></param>
        ///   <param name="length"></param>
        ///   <returns></returns>
        public static int ReadData(MemoryStream curStream, int startPosition, int length)
        {
            int result = -1;

            byte[] tempData = new byte[length];
            curStream.Position = startPosition;
            curStream.Read(tempData, 0, length);
            result = BitConverter.ToInt32(tempData, 0);

            return result;
        }

        ///   <summary>
        ///  Use Byte [] data to generate three-way BMP bitmap
        ///   </summary>
        ///   <param name="originalImageData"></param>
        ///   <param name="originalWidth"></param>
        ///   <param name="originalHeight"></param>
        ///   <returns></returns>
        public static Bitmap CreateBitmap(byte[] originalImageData, int originalWidth, int originalHeight, byte[] color_map)
        {
            Bitmap resultBitmap = new Bitmap(originalWidth, originalHeight, System.Drawing.Imaging.PixelFormat.Format8bppIndexed);

            MemoryStream curImageStream = new MemoryStream();
            resultBitmap.Save(curImageStream, System.Drawing.Imaging.ImageFormat.Bmp);
            curImageStream.Flush();

            int curPadNum = ((originalWidth * 8 + 31) / 32 * 4) - originalWidth;

            int bitmapDataSize = ((originalWidth * 8 + 31) / 32 * 4) * originalHeight;

            int dataOffset = ReadData(curImageStream, 10, 4);

            int paletteStart = 54;
            int paletteEnd = dataOffset;
            int color = 0;

            for (int i = paletteStart; i < paletteEnd; i += 4)
            {
                byte[] tempColor = new byte[4];
                tempColor[0] = (byte)color;
                tempColor[1] = (byte)color;
                tempColor[2] = (byte)color;
                tempColor[3] = (byte)0;
                color++;

                curImageStream.Position = i;
                curImageStream.Write(tempColor, 0, 4);
            }

            byte[] destImageData = new byte[bitmapDataSize];
            int destWidth = originalWidth + curPadNum;

            for (int originalRowIndex = originalHeight - 1; originalRowIndex >= 0; originalRowIndex--)
            {
                int destRowIndex = originalHeight - originalRowIndex - 1;

                for (int dataIndex = 0; dataIndex < originalWidth; dataIndex++)
                {
                    destImageData[destRowIndex * destWidth + dataIndex] = originalImageData[originalRowIndex * originalWidth + dataIndex];
                }
            }

            curImageStream.Position = dataOffset;

            curImageStream.Write(destImageData, 0, bitmapDataSize);

            curImageStream.Flush();

            resultBitmap = new Bitmap(curImageStream);

            resultBitmap = transForm8to24(resultBitmap, color_map);

            return resultBitmap;
        }

        // Implement Bitmap single channel to three-channel (split generation mask image (single channel) ==> RGB image)
        public static Bitmap transForm8to24(Bitmap bmp, byte[] color_map)
        {

            System.Drawing.Rectangle rect = new System.Drawing.Rectangle(0, 0, bmp.Width, bmp.Height);

            System.Drawing.Imaging.BitmapData bitmapData = bmp.LockBits(rect, System.Drawing.Imaging.ImageLockMode.ReadOnly, bmp.PixelFormat);

            int size8 = bitmapData.Stride * bmp.Height;
            byte[] grayValues = new byte[size8];

            Bitmap TempBmp = new Bitmap(bmp.Width, bmp.Height, PixelFormat.Format24bppRgb);
            BitmapData TempBmpData = TempBmp.LockBits(new Rectangle(0, 0, bmp.Width, bmp.Height), ImageLockMode.WriteOnly, PixelFormat.Format24bppRgb);

            int stride = TempBmpData.Stride;
            int offset = stride - TempBmp.Width;
            IntPtr iptr = TempBmpData.Scan0;
            int scanBytes = stride * TempBmp.Height;

            byte[] pixelValues = new byte[scanBytes];
            System.Runtime.InteropServices.Marshal.Copy(bitmapData.Scan0, grayValues, 0, size8);
            
            for (int i = 0; i < bmp.Height; i++)
            {

                for (int j = 0; j < bitmapData.Stride; j++)
                {

                    if (j >= bmp.Width)
                        continue;


                    int indexSrc = i * bitmapData.Stride + j;
                    int realIndex = i * TempBmpData.Stride + j * 3;

                    // color_id：The result of predicting
                    int color_id = (int)grayValues[indexSrc] % 256;

                    if (color_id == 0) // Segmentation Category 1 corresponds to 1, and the background is often 0, so it will be placed in [0, 0, 0] here.
                    {
                        pixelValues[realIndex] = 0;
                        pixelValues[realIndex + 1] = 0;
                        pixelValues[realIndex + 2] = 0;
                    }
                    else
                    {
                        pixelValues[realIndex] = color_map[color_id * 3];
                        pixelValues[realIndex + 1] = color_map[color_id * 3 + 1];
                        pixelValues[realIndex + 2] = color_map[color_id * 3 + 2];
                    }

                }

            }

            System.Runtime.InteropServices.Marshal.Copy(pixelValues, 0, iptr, scanBytes);
            TempBmp.UnlockBits(TempBmpData);
            bmp.UnlockBits(bitmapData);

            return TempBmp;
        }

        // RGB value collection of pseudo color map (color_map)
        private byte[] get_color_map_list(int num_classes = 256)
        {
            num_classes += 1;
            byte[] color_map = new byte[num_classes * 3];
            for (int i = 0; i < num_classes; i++)
            {
                int j = 0;
                int lab = i;
                while (lab != 0)
                {
                    color_map[i * 3] |= (byte)(((lab >> 0) & 1) << (7 - j));
                    color_map[i * 3 + 1] |= (byte)(((lab >> 1) & 1) << (7 - j));
                    color_map[i * 3 + 2] |= (byte)(((lab >> 2) & 1) << (7 - j));

                    j += 1;
                    lab >>= 3;
                }
            }

            color_map = color_map.Skip(3).ToArray();

            return color_map;
        }

        /// <summary>
        /// Get all files in the directory or specify file type file (contain all subfolders)
        /// </summary>
        /// <param name="path">Folder path</param>
        /// <param name="extName">Extension can be multiple eg: .mp3.wma.rm</param>
        /// <returns>List<FileInfo></returns>
        public static List<FileInfo> getFile(string path, string extName, List<FileInfo> lst)
        {
            try
            {
                DirectoryInfo fdir = new DirectoryInfo(path);
                FileInfo[] file = fdir.GetFiles();
                //FileInfo[] file = Directory.GetFiles(path);
                if (file.Length != 0)
                {
                    foreach (FileInfo f in file)
                    {
                        if (extName.ToLower().IndexOf(f.Extension.ToLower()) >= 0)
                        {
                            lst.Add(f);
                        }
                    }
                }
                return lst;
            }
            catch (Exception ex)
            {
                throw ex;
            }
        }

        // Convert BTIMAP class to Byte [] class function
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

        // Creates an IntPtr pointer to a float array type
        public static IntPtr FloatToIntptr(float[] bytes)
        {
            GCHandle hObject = GCHandle.Alloc(bytes, GCHandleType.Pinned);
            return hObject.AddrOfPinnedObject();
        }

        // Check if the Maskrcnn model is started on the GPU - only supports GPU reasoning, because the memory takes up, 
        // the CPU may overflow, resulting in unable to construct
        public static bool CheckMaskRCNN_workOnGpu(string model_type, bool use_gpu)
        {
            if (model_type == "mask")
            {
                if (use_gpu == false)
                    return false;
            }
            return true;
        }
    }
}
