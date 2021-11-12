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
        /*****************          1.推理DLL导入实现          ****************/
        /**********************************************************************/
        // 加载推理相关方法
        [DllImport("model_infer.dll", EntryPoint = "InitModel")] // 模型统一初始化方法: 需要yml、pdmodel、pdiparams
        public static extern void InitModel(string model_type, string model_filename, string params_filename, string cfg_file, bool use_gpu, int gpu_id, ref byte paddlex_model_type);

        [DllImport("model_infer.dll", EntryPoint = "Det_ModelPredict")]  // PaddleDetection模型推理方法
        public static extern void Det_ModelPredict(byte[] img, int W, int H, int C, IntPtr output, int[] BoxesNum, ref byte label);

        [DllImport("model_infer.dll", EntryPoint = "Seg_ModelPredict")]  // PaddleSeg模型推理方法
        public static extern void Seg_ModelPredict(byte[] img, int W, int H, int C, ref byte output);

        [DllImport("model_infer.dll", EntryPoint = "Cls_ModelPredict")]  // PaddleClas模型推理方法
        public static extern void Cls_ModelPredict(byte[] img, int W, int H, int C, ref float score, ref byte category, ref int category_id);

        [DllImport("model_infer.dll", EntryPoint = "Mask_ModelPredict")]  // Paddlex的MaskRCNN模型推理方法
        public static extern void Mask_ModelPredict(byte[] img, int W, int H, int C, IntPtr output, ref byte Mask_output, int[] BoxesNum, ref byte label);

        [DllImport("model_infer.dll", EntryPoint = "DestructModel")]  // 分割、检测、识别模型销毁方法
        public static extern void DestructModel();


        /**********************************************************************/
        /******************         2.控制参数的声明          *****************/
        /**********************************************************************/
        // 模型基本参数
        string imgfile = null; // 推理的图片路径 -- 单张图片路径
        List<string> imgfiles = new List<string>(); // 推理的图片路径 -- 多张图片路径
        string videofile = null; // 推理的视频路径
        bool use_gpu = false;  // 是否使用gpu
        int gpu_id = 0;  // 默认GPU_ID为0
        float det_threshold = 0.5F; // 设置阈值 -- 默认0.5
        string model_type = "det"; // 模型类型 -- 检测: det / paddlex
        string model_filename = null; // *.pdmodel -- 模型文件
        string params_filename = null;  // *.pdiparams == 参数文件
        string cfg_file = null;  // *.yml -- 配置文件

        // paddlex模型下的实际模型类型
        byte[] paddlex_model_type = new byte[10];  // det/seg/clas + \0
        // 记录paddlex模式存在
        bool paddlex_doing = false;
        // 图片类型
        string[] img_type = {"jpg", "png", "JPEG", "jpeg"};

        // 模型已完成初始化的标志
        static int has_model_init = 0;
        // 是否正在进行推理预测
        static int is_infer = 0;
        // 是否中断推理
        static int isBreakInfer = 0;
        // 推理线程进行标志
        static int infer_one_img_flag = 0;
        static int infer_many_img_flag = 0;
        static int infer_video_img_flag = 0;
        // 连续推理的间隔ms
        int continue_infer_delay = 50;


        /**********************************************************************/
        /**************         3.窗体加载与关闭的实现          ***************/
        /**********************************************************************/
        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            comboBox1.SelectedIndex = 0;  // 初始运行环境  -- cpu
            comboBox2.SelectedIndex = 0;  // 初始模型 -- det
            comboBox3.SelectedIndex = 5;  // 初始阈值 -- 0.5
            numericUpDown1.Value = 50;    // 设置初始连续推理间隔为50ms

            label7.Text = "0.00"; // 默认推理耗时
            textBox1.Text = "0"; // 默认GPU_ID为0
        }

        private void Form1_FromClosing(object sender, EventArgs e)
        {
            while (is_infer != 0) // 有推理进程在运行
            {
                isBreakInfer = 1; // 关掉进程
            } // 等待推理进程完全结束

            if (has_model_init == 1) // 有初始化的模型存在，销毁模型后正常退出
            {
                DestructModel(); // 销毁模型
            }
        }


        /**********************************************************************/
        /*****************          4.细节参数选项实现          ***************/
        /**********************************************************************/
        int comboBox_clicked = 0;
        int comboBox1_last_index = 0;

        // 选择运行环境 - 是否使用GPU
        private void comboBox1_SelectedIndexChanged(object sender, EventArgs e)
        {

            // 推理过程中，不支持运行环境选择
            if (is_infer == 1 && comboBox_clicked == 0)
            {
                MessageBox.Show("正在推理中，请推理完成后再选择运行环境重新初始化!", "提示", MessageBoxButtons.OK, MessageBoxIcon.Error);
                comboBox_clicked = 1;
                comboBox1.SelectedIndex = comboBox1_last_index;  // 使用上一次改变后得到的index
                return;
            }

            // 已经初始化，不支持运行环境选择
            if (has_model_init == 1 && comboBox_clicked == 0)
            {
                MessageBox.Show("模型已初始化，请销毁模型后再进行运行环境选择!\n(CPU,GPU)", "提示", MessageBoxButtons.OK, MessageBoxIcon.Error);
                comboBox_clicked = 1;
                comboBox1.SelectedIndex = comboBox1_last_index;  // 使用上一次改变后得到的index
                return;
            }

            // 对应两种运行环境
            if (comboBox1.SelectedItem.ToString() == "GPU") // 使用gpu
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

        // 修改GPU_ID -- 指定gpu
        int last_gpu_id = 0;
        int gpu_id_done = 0;
        private void textBox1_TextChanged(object sender, EventArgs e)
        {
            // 推理过程中，不支持GPU指定 -- 未定义操作
            if (is_infer == 1 && gpu_id_done == 0)
            {
                MessageBox.Show("正在推理中，请推理完成后再选择指定GPU重新初始化!", "提示", MessageBoxButtons.OK, MessageBoxIcon.Error);
                gpu_id_done = 1;
                gpu_id = last_gpu_id;
                textBox1.Text = $"{gpu_id}";
                return;
            }

            // 已经初始化，不支持GPU指定 -- 未定义操作
            if (has_model_init == 1 && gpu_id_done == 0)
            {
                MessageBox.Show("模型已初始化，请销毁模型后再进行GPU指定!\n(GPU:x)", "提示", MessageBoxButtons.OK, MessageBoxIcon.Error);
                gpu_id_done = 1;
                gpu_id = last_gpu_id;
                textBox1.Text = $"{gpu_id}";
                return;
            }

            if (gpu_id_done == 0) // 定义操作下的修改才会执行以下内容
            {
                string gpu_id_str = textBox1.Text.ToString();
                if (gpu_id_str.Length != 0)
                {
                    last_gpu_id = gpu_id;
                    try
                    {
                        gpu_id = Int32.Parse(gpu_id_str);  // 获取新的GPU_id
                    }
                    catch (Exception ex)
                    {
                        gpu_id = last_gpu_id;
                        textBox1.Text = $"{gpu_id}";
                        MessageBox.Show("GPU_ID只能输入数字!");
                    }
                }
            }

            gpu_id_done = 0; // 复原状态值
        }


        int comboBox2_last_index = 0;

        // 执行推理的模型的类型选择
        private void comboBox2_SelectedIndexChanged(object sender, EventArgs e)
        {

            // 推理过程中，不支持模型类型选择
            if (is_infer == 1 && comboBox_clicked == 0)
            {
                MessageBox.Show("正在推理中，请推理完成后再选择模型类型重新初始化!", "提示", MessageBoxButtons.OK, MessageBoxIcon.Error);
                comboBox_clicked = 1;
                comboBox2.SelectedIndex = comboBox2_last_index;  // 使用上一次改变后得到的index
                return;
            }

            // 已经初始化，发出警告，提示重新初始化，模型类型的修改才会生效
            if (has_model_init == 1 && comboBox_clicked == 0)
            {
                MessageBox.Show("模型已初始化，请销毁模型后再进行模型类型选择!\n(det,seg,clas)", "提示", MessageBoxButtons.OK, MessageBoxIcon.Error);
                comboBox_clicked = 1;
                comboBox2.SelectedIndex = comboBox2_last_index;  // 使用上一次改变后得到的index
                return;
            }

            // 对应三种类型
            if (comboBox2.SelectedItem.ToString() == "det")  // 加载检测模型  -- 推理已实现
            {
                model_type = comboBox2.SelectedItem.ToString();
                paddlex_doing = false;  // 进入非paddlex模式 -- 检测
            }
            else if (comboBox2.SelectedItem.ToString() == "seg") // 加载分割模型 -- 推理已实现
            {
                model_type = comboBox2.SelectedItem.ToString();
                paddlex_doing = false;  // 进入非paddlex模式 -- 分割
            }
            else if (comboBox2.SelectedItem.ToString() == "clas") // 加载识别模型 -- 推理已实现
            {
                model_type = comboBox2.SelectedItem.ToString();
                paddlex_doing = false;  // 进入非paddlex模式 -- 识别
            }
            else if (comboBox2.SelectedItem.ToString() == "mask") // 加载实例分割MaskRCNN模型 -- 推理已实现
            {
                model_type = comboBox2.SelectedItem.ToString();
                paddlex_doing = false;  // 进入非paddlex模式 -- 实则也为paddlex导出的模型
            }
            else if (comboBox2.SelectedItem.ToString() == "paddlex") // 加载识别模型 -- 推理已实现
            {
                model_type = comboBox2.SelectedItem.ToString();
                // 重复选中paddlex，不修改状态
            }
            comboBox2_last_index = comboBox2.SelectedIndex;

            
            comboBox_clicked = 0;
        }

        
        int comboBox3_last_index = 0;
        // 设置目标检测的检测阈值
        private void comboBox3_SelectedIndexChanged(object sender, EventArgs e)
        {
            // 推理过程中，不支持检测阈值选择
            if (is_infer == 1 && comboBox_clicked == 0)
            {
                MessageBox.Show("正在推理中，检测阈值修改将在本次模型推理完成后生效!", "提示", MessageBoxButtons.OK, MessageBoxIcon.Error);

                comboBox_clicked = 1;
                comboBox3.SelectedIndex = comboBox3_last_index;  // 使用上一次改变后得到的index
                return;
            }


            // 修改检测阈值
            det_threshold = float.Parse(comboBox3.SelectedItem.ToString());
            comboBox3_last_index = comboBox3.SelectedIndex;  // 保存本次的索引


            comboBox_clicked = 0;
        }


        // 连续推理的间隔时间长度 -- 图片文件夹推理
        private void numericUpDown1_ValueChanged(object sender, EventArgs e)
        {
            // 配置连续推理的延时
            continue_infer_delay = ((int)numericUpDown1.Value);
        }


        /**********************************************************************/
        /*****************          5.选择控制组件实现          ***************/
        /**********************************************************************/
        // 加载模型相关文件的文件夹 -- 测试完成
        private void button1_Click(object sender, EventArgs e)
        {
            // 推理过程中，不支持模型初始化
            if (is_infer == 1)
            {
                MessageBox.Show("正在推理中，请推理完成后再初始化加载模型!", "提示", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            // 先检查MaskRCNN启动状态
            if (!CheckMaskRCNN_workOnGpu(model_type, use_gpu)) // MaskRCNN环境不在GPU，则报错提醒
            {
                MessageBox.Show("MaskRCNN推理仅支持GPU环境，请重新选择启动环境!\n(因为CPU环境可能存在内存不足，导致推理失败。)", "提示", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            int dir_load_flag = 0;  // 文件夹选择的标志位
            string dir_path = null;
            folderBrowserDialog1.Description = "请选择模型文件夹";
            DialogResult folder = folderBrowserDialog1.ShowDialog();
            if (folder == DialogResult.OK || folder == DialogResult.Yes)
            {
                dir_path = folderBrowserDialog1.SelectedPath;
                if (string.IsNullOrEmpty(dir_path)) // 判断是否选择了模型文件
                {
                    MessageBox.Show("请选择模型路径/模型路径为空!", "提示", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    dir_load_flag = 0;
                }
                else
                {
                    dir_load_flag = 1;
                }
            }

            if (dir_load_flag == 1) // 寻找模型文件
            {
                List<FileInfo> model_lst = new List<FileInfo>();
                List<FileInfo> params_lst = new List<FileInfo>();
                List<FileInfo> cfg_lst = new List<FileInfo>();
                model_lst = getFile(dir_path, ".pdmodel", model_lst); // 返回匹配的文件
                params_lst = getFile(dir_path, ".pdiparams", params_lst); // 返回匹配的文件
                cfg_lst = getFile(dir_path, ".yml", cfg_lst); // 返回匹配的文件
                if (cfg_lst.Count == 0)
                    cfg_lst = getFile(dir_path, ".yaml", cfg_lst); // 返回匹配的文件

                if (model_lst.Count != 1 || params_lst.Count != 1)
                {
                    MessageBox.Show("模型文件加载失败!\n请注意模型文件夹下应包含以下文件各一个:\n*.pdmodel, *.pdiparams", "提示");
                    return;
                }

                model_filename = model_lst[0].FullName;
                params_filename = params_lst[0].FullName;

                if (cfg_lst.Count == 0)  // 没有yml文件
                {
                    MessageBox.Show("模型文件加载失败!\n请注意模型文件夹下应包含以下文件:\nmodel.yml/model.yaml", "提示");
                    return;
                }
                else if (cfg_lst.Count > 2) // yml过多， > 2 
                {
                    MessageBox.Show("模型文件加载失败!\n请注意模型文件夹下应至多包含以下文件(yml文件个数不得超过2):\nmodel.yml/model.yaml，pipeline.yml/pipeline.yaml", "提示");
                    return;
                }
                else if (cfg_lst.Count == 2) // 对于包含多个yml文件的情况的处理 ， == 2, 及针对paddlex的处理
                {
                    // 筛选yml文件 -- 只需要model.yml
                    for (int i = 0; i < 2; i++)
                    {
                        if (cfg_lst[i].Name == "model.yml" || cfg_lst[i].Name == "model.yaml")
                        {
                            cfg_file = cfg_lst[i].FullName;
                            break;
                        }
                    }
                }
                else if (cfg_lst.Count == 1)  // 直接取出唯一的yml文件
                {
                    cfg_file = cfg_lst[0].FullName;
                }

                int raise_ex_flag = 0;  // 是否发生了异常
                int is_Mask = 0;  // 当前初始化模型是否未MaskRCNN
                if (has_model_init == 1) // 已经初始化，再次初始化前要完成上一个模型的销毁
                {
                    // 销毁模型
                    DestructModel();

                    // 初始化模型
                    try
                    {
                        // 保持paddlex模式
                        if (paddlex_doing == true) model_type = "paddlex";
                        if (model_type == "mask")
                        {
                            model_type = "paddlex"; // 因为MaskRCNN来自paddlex训练，所以这里先转未paddlex
                            is_Mask = 1;
                        }
                        InitModel(model_type, model_filename, params_filename, cfg_file, use_gpu, gpu_id, ref paddlex_model_type[0]);

                        if (is_Mask == 1) // 初始化完成后还原model_type
                        {
                            model_type = "mask";
                        }
                        if (model_type == "paddlex") // 如果当前初始模型类型为paddlex，则初始化完成后，转为paddlex模型的实际类型
                        {
                            paddlex_doing = true; // 进入paddlex类型模式
                            model_type = System.Text.Encoding.UTF8.GetString(paddlex_model_type).Split('\0')[0]; // 得到实际运行的模型类型 -- Split去掉多余的\0（原byte[]长度为10，有许多多余的\0）
                        }
                    }
                    catch (Exception ex)
                    {
                        raise_ex_flag = 1;  // 发生了异常
                        MessageBox.Show("1.请确定文件中包含有效的模型文件(*.pdmodel, *.pdiparams, *.yml)!\n2.请检查模型文件与模型类型是否一致!\n3.其它原因:GPU号有误，yml中预处理有误...", "模型初始化失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    }
                }
                else
                {
                    // 初始化模型
                    try
                    {
                        // 保持paddlex模式
                        if (paddlex_doing == true) model_type = "paddlex";
                        if (model_type == "mask")
                        {
                            model_type = "paddlex"; // 因为MaskRCNN来自paddlex训练，所以这里先转未paddlex
                            is_Mask = 1;
                        }
                        InitModel(model_type, model_filename, params_filename, cfg_file, use_gpu, gpu_id, ref paddlex_model_type[0]);

                        if (is_Mask == 1) // 初始化完成后还原model_type
                        {
                            model_type = "mask";
                        }
                        if (model_type == "paddlex") // 如果当前初始模型类型为paddlex，则初始化完成后，转为paddlex模型的实际类型
                        {
                            paddlex_doing = true; // 进入paddlex类型模式

                            model_type = System.Text.Encoding.UTF8.GetString(paddlex_model_type).Split('\0')[0]; // 得到实际运行的模型类型
                        }
                    }
                    catch (Exception ex)
                    {
                        raise_ex_flag = 1;  // 发生了异常
                        MessageBox.Show("1.请确定文件中包含有效的模型文件(*.pdmodel, *.pdiparams, *.yml)!\n2.请检查模型文件与模型类型是否一致!\n3.其它原因:GPU号有误，yml中预处理有误...", "模型初始化失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    }
                }

                if (raise_ex_flag == 0) // 未发生异常时，才进行正常的运行提示
                {
                    has_model_init = 1;  // 已经完成初始化 
                    if (use_gpu)
                    {
                        MessageBox.Show($"模型文件已加载到GPU:{gpu_id}!\n(模型类型为: {model_type.Split('\0')[0]})", "提示");
                    }
                    else
                    {
                        MessageBox.Show($"模型类型为: {model_type.Split('\0')[0]}", "提示");
                    }
                    button1.Text = "模型已初始化";  // 更改按键提示信息
                }
            }
        }

        // 加载单张图片 -- 测试完成
        private void button2_Click(object sender, EventArgs e)
        {
            // 推理过程中，不支持推理数据集选择与加载
            if (is_infer == 1)
            {
                MessageBox.Show("正在推理中，请推理完成后再选择推理数据!", "提示", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            int has_load_img_file_flag = 1;
            openFileDialog1.Filter = "(*.png;*.jpg;*.JPEG;*.jpeg)|*.*"; // 设置打开的文件类型
            DialogResult dr = openFileDialog1.ShowDialog();
            //获取所打开文件的文件名
            string filename = openFileDialog1.FileName;
            if (dr != System.Windows.Forms.DialogResult.OK || string.IsNullOrEmpty(filename)) // 检验文件是否选择成功
            {
                has_load_img_file_flag = 0; // 没有读取到图片路径
            }

            if (has_load_img_file_flag==1)  // 正确加载图片才有以下执行
            {
                // 划分文件，获取后缀
                string[] final_tag = filename.Split('.');
                int flag = 0;
                foreach (string type in img_type)
                {
                    if (final_tag[1] == type)
                    {
                        flag = 1;  // 类型满足预置图片类型时，flag为1

                        imgfile = filename; // 单张图片

                        // 显示加载的图片
                        pictureBox1.Image = Image.FromFile(imgfile);
                        pictureBox1.SizeMode = PictureBoxSizeMode.Zoom;

                        // 清空其它推理数据的来源
                        videofile = null;
                        imgfiles.Clear(); // 加载单张图片，清空文件夹图片索引

                        MessageBox.Show("图片加载完成!", "提示");

                        button2.Text = "图片已加载";  // 更改按键提示信息
                        button3.Text = "加载图片文件夹";  // 更改按键提示信息
                        button4.Text = "加载视频流";  // 更改按键提示信息
                    }
                }
            }
        }

        // 加载文件夹图片
        private void button3_Click(object sender, EventArgs e)
        {
            // 推理过程中，不支持推理数据集选择与加载
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

            if (img_dir_load_flag == 1) // 读取文件夹中的指定图片文件
            {
                List<FileInfo> lst = new List<FileInfo>();
                lst = getFile(img_dir_path, ".jpg", lst); // 返回匹配的文件
                lst = getFile(img_dir_path, ".png", lst); // 返回匹配的文件
                lst = getFile(img_dir_path, ".JPEG", lst); // 返回匹配的文件

                foreach (FileInfo Image_File in lst)  // 添加文件
                {
                    imgfiles.Add(Image_File.FullName);
                }

                if (imgfiles.Count == 0)
                {
                    MessageBox.Show("请输入选择非空/包含正确图片类型的图片文件夹!\n(*.png, *.jpg, *.JPEG)", "图片解析失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;  // 提前终止该函数操作 -- 保留原有加载数据
                }

                // 展示第一张图片
                // 显示加载的图片
                pictureBox1.Image = Image.FromFile(imgfiles[0]);
                pictureBox1.SizeMode = PictureBoxSizeMode.Zoom;

                // 清空其它推理数据的来源
                imgfile = null; // 既然加载文件夹，则单张图片的索引应该清空
                videofile = null;
                MessageBox.Show("图片文件夹加载完成!", "提示");

                button3.Text = "图片文件夹已加载";  // 更改按键提示信息
                button2.Text = "加载图片";  // 更改按键提示信息
                button4.Text = "加载视频流";  // 更改按键提示信息
            }
       
        }

        // 加载视频流
        private void button4_Click(object sender, EventArgs e)
        {
            // 推理过程中，不支持模型初始化
            if (is_infer == 1)
            {
                MessageBox.Show("正在推理中，请推理完成后再选择推理数据!", "提示", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            int has_load_mp4_file_flag = 1;
            openFileDialog1.Filter = "(*.mp4)|*.*"; // 设置打开的文件类型
            DialogResult dr = openFileDialog1.ShowDialog();
            //获取所打开文件的文件名
            string filename = openFileDialog1.FileName;
            if (dr != System.Windows.Forms.DialogResult.OK || string.IsNullOrEmpty(filename)) // 检验文件是否选择成功
            {
                if (string.IsNullOrEmpty(filename)) MessageBox.Show("请选择视频(*.mp4)文件!", "视频路径为空", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                has_load_mp4_file_flag = 0;
            }
            
            if (has_load_mp4_file_flag==1)  // 读取到视频mp4
            {
                // 划分文件，获取后缀
                string[] final_tag = filename.Split('.');

                if (final_tag[1] == "mp4")
                {
                    videofile = filename; // mp4视频路径

                    Bitmap image = null;
                    Mat frame = new Mat();
                    VideoCapture capture = new VideoCapture(); // 创建一个摄像头
                    capture.Open(videofile);
                    bool read_success = capture.Read(frame);   // 帧是否读取成功
                    if (!read_success)
                    {
                        MessageBox.Show("无法读取视频的帧！！！", "提示");
                    }
                    else
                    {
                        image = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(frame);

                        // 显示加载的视频的第一帧
                        pictureBox1.Image = image;
                        pictureBox1.SizeMode = PictureBoxSizeMode.Zoom;

                        capture = null;  // 收回内存
                        frame = null;    // 收回内存
                        image = null;    // 收回内存

                        // 清空其它推理数据的来源
                        imgfile = null;
                        imgfiles.Clear(); // 加载单张图片，清空文件夹图片索引
                        MessageBox.Show("视频加载完成!", "提示");

                        button4.Text = "视频流已加载";  // 更改按键提示信息
                        button2.Text = "加载图片";  // 更改按键提示信息
                        button3.Text = "加载图片文件夹";  // 更改按键提示信息
                    }
                }
                else
                {
                    // 保持原有数据加载情况，并发出错误警告
                    MessageBox.Show("请选择mp4视频文件!", "视频资源加载失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }

            }
        }

        // 执行推理
        private void button5_Click(object sender, EventArgs e)
        {
            // has_model_init: 确保模型已经初始化
            if (imgfile != null && is_infer==0 && has_model_init == 1)  // 单张图片的预测 -- is_infer 等于 0， 表示没有任何进程在运行
            {
                Thread infer_one_img_thread = null;
                if (model_type == "det") infer_one_img_thread = new Thread(new ThreadStart(delegate { det_infer_one_img();  }));
                else if (model_type == "seg") infer_one_img_thread = new Thread(new ThreadStart(delegate { seg_infer_one_img(); }));
                else if (model_type == "clas") infer_one_img_thread = new Thread(new ThreadStart(delegate { cls_infer_one_img(); }));
                else if (model_type == "mask") infer_one_img_thread = new Thread(new ThreadStart(delegate { mask_infer_one_img(); }));
                MessageBox.Show("开始图片推理任务!", "提示");
                infer_one_img_thread.Start(); // 启动任务
                infer_one_img_flag = 1;  // 标志着图片正在推理执行

            }
            else if (imgfiles.Count != 0 && is_infer == 0 && has_model_init == 1) // 图片文件夹的预测
            {
                Thread infer_many_img_thread = null;
                if (model_type == "det") infer_many_img_thread = new Thread(new ThreadStart(delegate { det_infer_many_img(); }));
                else if (model_type == "seg") infer_many_img_thread = new Thread(new ThreadStart(delegate { seg_infer_many_img(); }));
                else if (model_type == "clas") infer_many_img_thread = new Thread(new ThreadStart(delegate { cls_infer_many_img(); }));
                else if (model_type == "mask") infer_many_img_thread = new Thread(new ThreadStart(delegate { mask_infer_many_img(); }));
                MessageBox.Show("开始图片文件夹推理任务!", "提示");
                infer_many_img_thread.Start(); // 启动任务
                infer_many_img_flag = 1;  // 标志着图片文件夹正在推理执行
            }
            else if (videofile != null && is_infer == 0 && has_model_init == 1)
            {
                Thread infer_video_img_thread = null;
                if (model_type == "det") infer_video_img_thread = new Thread(new ThreadStart(delegate { det_infer_video_img(); }));
                else if (model_type == "seg") infer_video_img_thread = new Thread(new ThreadStart(delegate { seg_infer_video_img(); }));
                else if (model_type == "clas") infer_video_img_thread = new Thread(new ThreadStart(delegate { cls_infer_video_img(); }));
                else if (model_type == "mask") infer_video_img_thread = new Thread(new ThreadStart(delegate { mask_infer_video_img(); }));
                MessageBox.Show("开始视频推理任务!", "提示");
                infer_video_img_thread.Start(); // 启动任务
                infer_video_img_flag = 1;  // 标志着视频正在推理执行
            }
            else if (is_infer == 1 && has_model_init == 1)
            {
                if (infer_one_img_flag == 1) MessageBox.Show("正在进行推理任务!", "请勿再执行图片推理任务", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                if (infer_many_img_flag == 1) MessageBox.Show("正在进行推理任务!", "请勿再执行图片文件夹推理任务", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                if (infer_video_img_flag == 1) MessageBox.Show("正在进行推理任务!", "请勿再执行视频推理任务", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
            
            if (has_model_init == 0 && (imgfile == null && imgfiles.Count == 0 && videofile == null)) // 模型未初始化，数据未加载
            {
                MessageBox.Show("请先初始化模型，并选择加载的推理数据后，再点击模型推理!", "推理执行失败", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
            else if (has_model_init == 0 && (imgfile != null || imgfiles.Count != 0 || videofile != null)) // 模型未初始化，数据加载
            {
                MessageBox.Show("请初始化模型，再点击模型推理!", "推理执行失败", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
            else if (has_model_init != 0 && (imgfile == null && imgfiles.Count == 0 && videofile == null)) // 模型初始化，数据未加载
            {
                MessageBox.Show("请选择加载的推理数据，再点击模型推理!", "推理执行失败", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
        }

        // 终止推理
        private void button6_Click(object sender, EventArgs e)
        {
            isBreakInfer = 1;  // 发出推理终止的信号 -- 线程会开始终止(非kill终止)
        }

        // 销毁已初始化好的模型
        private void button7_Click(object sender, EventArgs e)
        {
            if (is_infer == 0)
            {
                if (has_model_init == 1)
                {
                    // 销毁模型
                    try // 进行未定义的模型销毁时的异常处理
                    {
                        DestructModel();
                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show("当前未初始化模型，无需销毁!", "模型销毁失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    }
                    has_model_init = 0;
                }

                button1.Text = "初始化模型"; // 重置按键状态
            }
            else
            {
                MessageBox.Show("请先中断模型推理，再销毁已初始化的模型!", "提示", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }


        /***********************************************************************/
        /*****************          6.可视化推理实现部分          **************/
        /***********************************************************************/
        // 检测单张图片
        private void det_infer_one_img()
        {
            is_infer = 1; // 进入推理

            

            byte[] color_map = get_color_map_list(256);

            //Bitmap bmp = new Bitmap(imgfile);
            Bitmap bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(Cv2.ImRead(imgfile));
            byte[] inputData = GetBGRValues(bmp, out int stride);

            float[] resultlist = new float[600];
            IntPtr results = FloatToIntptr(resultlist);
            int[] boxesInfo = new int[1]; // 10 boundingbox
            byte[] labellist = new byte[1000];    //新建字节数组：label1_str label2_str 

            int raise_ex_flag = 0;  // 是否发生了异常
            try
            {
                TimeSpan infer_start_time = new TimeSpan(DateTime.Now.Ticks);
                // 第四个参数为输入图像的通道数
                Det_ModelPredict(inputData, bmp.Width, bmp.Height, 3, results, boxesInfo, ref labellist[0]);
                TimeSpan infer_end_time = new TimeSpan(DateTime.Now.Ticks);

                string strGet = System.Text.Encoding.Default.GetString(labellist, 0, labellist.Length);    //将字节数组转换为字符串
                string[] predict_Label_List = strGet.Split(' ');  // 预测的类别情况
                                                                  // MessageBox.Show($"Box_Number: {boxesInfo[0]}");
                using OpenCvSharp.Mat mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(bmp);//用bitmap转换为mat
                for (int i = 0; i < boxesInfo[0]; i++) // 未绘制图像
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

                        // 获取文本区域的大小
                        var text_size = Cv2.GetTextSize($"{predict_Label_List[i]}-{labelindex}-{score:f2}",
                                         HersheyFonts.HersheySimplex, 1, 2, out int baseline);  // 1倍大小的HersheySimplex，高度为22
                                                                                                // 获取文本区域的左下顶点 -- 右上角
                        int left_down_x = (int)left + 22; // 小偏移调整量: (int)(text_size.Width/10)
                        int left_down_y = (int)top + text_size.Height;

                        // 绘制矩形，书写类别
                        Cv2.Rectangle(mat, new OpenCvSharp.Rect((int)left, (int)top, (int)right, (int)down), new OpenCvSharp.Scalar(color_[0], color_[1], color_[2]), 2, LineTypes.AntiAlias);//LineTypes.AntiAlias:反锯齿效果
                        Cv2.PutText(mat, $"{predict_Label_List[i]}-{labelindex}-: {score:f2}", new OpenCvSharp.Point(left_down_x, left_down_y), HersheyFonts.HersheySimplex, 1, new OpenCvSharp.Scalar(color_[0], color_[1], color_[2]), 2, LineTypes.Link4);
                    }
                }

                // 转换回bitmap进行显示
                bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(mat);
                // 反馈到另一个picturebox上
                if (pictureBox2.Image != null) pictureBox2.Image.Dispose();
                pictureBox2.Image = bmp;
                pictureBox2.SizeMode = PictureBoxSizeMode.Zoom;

                // 展示推理耗时
                TimeSpan start2end_time = infer_end_time.Subtract(infer_start_time).Duration();
                double cost_milliseconds = start2end_time.TotalMilliseconds;
                // 通过委托展示到label上
                Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };//定义一个委托
                label7.Invoke(AsyncUIDelegate, new object[] { $"{cost_milliseconds:f2}" });
            }
            catch (Exception e)
            {
                raise_ex_flag = 1;

                // 默认耗时为0ms
                Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };//定义一个委托
                label7.Invoke(AsyncUIDelegate, new object[] { "0.00" });
                MessageBox.Show("1.请检查模型文件与模型类型是否一致!\n2.内存溢出，yml预处理有误，图片格式确保为1/3通道...", "模型运行失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

            isBreakInfer = 0; // 清空标志
            is_infer = 0; // 退出推理  -- 解除推理状态
            infer_one_img_flag = 0; // 重置当前推理状态 -- 解除图片推理状态
            if (raise_ex_flag == 0) MessageBox.Show("图片推理完成!"); // 未发生异常，正常显示推理完成提示
        }

        // 检测图片文件夹
        private void det_infer_many_img()
        {
            is_infer = 1; // 进入推理

            byte[] color_map = get_color_map_list(256);

            int raise_ex_flag = 0;  // 是否发生了异常
            try
            {
                foreach (string img_file in imgfiles)
                {
                    if (isBreakInfer == 1) break;  // 中断推理

                    Bitmap show_image = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(Cv2.ImRead(img_file));

                    Bitmap bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(Cv2.ImRead(img_file));
                    byte[] inputData = GetBGRValues(bmp, out int stride);

                    float[] resultlist = new float[600];
                    IntPtr results = FloatToIntptr(resultlist);
                    int[] boxesInfo = new int[1];
                    byte[] labellist = new byte[1000];    //新建字节数组

                    TimeSpan infer_start_time = new TimeSpan(DateTime.Now.Ticks);
                    //第四个参数为输入图像的通道数
                    Det_ModelPredict(inputData, bmp.Width, bmp.Height, 3, results, boxesInfo, ref labellist[0]);
                    TimeSpan infer_end_time = new TimeSpan(DateTime.Now.Ticks);

                    string strGet = System.Text.Encoding.Default.GetString(labellist, 0, labellist.Length);    //将字节数组转换为字符串
                    string[] predict_Label_List = strGet.Split(' ');  // 预测的类别情况
                                                                      //MessageBox.Show($"Box_Number: {boxesInfo[0]}");
                                                                      //Console.WriteLine("labellist: {0}", strGet);
                                                                      // 转换为mat数据，方便opencv处理
                    using OpenCvSharp.Mat mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(bmp);//用bitmap转换为mat
                    for (int i = 0; i < boxesInfo[0]; i++) // 未绘制图像
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

                            // 获取文本区域的大小
                            var text_size = Cv2.GetTextSize($"{predict_Label_List[i]}-{labelindex}-{score:f2}",
                                             HersheyFonts.HersheySimplex, 1, 2, out int baseline);  // 1倍大小的HersheySimplex，高度为22
                                                                                                    // 获取文本区域的左下顶点 -- 右上角
                            int left_down_x = (int)left + 22; // 小偏移调整量: (int)(text_size.Width/10)
                            int left_down_y = (int)top + text_size.Height;

                            // 绘制矩形，书写类别
                            Cv2.Rectangle(mat, new OpenCvSharp.Rect((int)left, (int)top, (int)right, (int)down), new OpenCvSharp.Scalar(color_[0], color_[1], color_[2]), 2, LineTypes.AntiAlias);//LineTypes.AntiAlias:反锯齿效果
                            Cv2.PutText(mat, $"{predict_Label_List[i]}-{labelindex}-: {score:f2}", new OpenCvSharp.Point(left_down_x, left_down_y), HersheyFonts.HersheySimplex, 1, new OpenCvSharp.Scalar(color_[0], color_[1], color_[2]), 2, LineTypes.Link4);
                        }
                    }
                    // 转换回bitmap进行显示
                    bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(mat);

                    // 显示图片
                    if (pictureBox1.Image != null) pictureBox1.Image.Dispose();
                    pictureBox1.Image = show_image; //显示原始图片到box1
                    pictureBox1.SizeMode = PictureBoxSizeMode.Zoom; //显示原始图片到box1
                    // 反馈到另一个picturebox上
                    if (pictureBox2.Image != null) pictureBox2.Image.Dispose();
                    pictureBox2.Image = bmp;
                    pictureBox2.SizeMode = PictureBoxSizeMode.Zoom;

                    // 展示推理耗时
                    TimeSpan start2end_time = infer_end_time.Subtract(infer_start_time).Duration();
                    double cost_milliseconds = start2end_time.TotalMilliseconds;
                    // 通过委托展示到label上
                    Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };//定义一个委托
                    label7.Invoke(AsyncUIDelegate, new object[] { $"{cost_milliseconds:f2}" });

                    Thread.Sleep(continue_infer_delay);  // 连续识别时，每张图片间隔continue_infer_delay毫秒
                }
            }
            catch (Exception e)
            {
                raise_ex_flag = 1;

                // 默认耗时为0ms
                Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };//定义一个委托
                label7.Invoke(AsyncUIDelegate, new object[] { "0.00" });
                MessageBox.Show("1.请检查模型文件与模型类型是否一致!\n2.内存溢出，yml预处理有误，图片格式确保为1/3通道...", "模型运行失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

            isBreakInfer = 0; // 清空标志

            // DestructModel(); // 销毁模型

            is_infer = 0; // 退出推理  -- 解除推理状态
            infer_many_img_flag = 0;  // 重置当前推理状态 -- 解除图片推理状态
            if (raise_ex_flag == 0) MessageBox.Show("图片文件夹推理完成!"); // 未发生异常，正常显示推理完成提示
        }

        // 检测视频流
        private void det_infer_video_img()
        {
            is_infer = 1; // 进入推理

            byte[] color_map = get_color_map_list(256);

            VideoCapture capture = new VideoCapture();
            capture.Open(videofile); // 读取视频

            using Mat frame = new Mat();

            int raise_ex_flag = 0;  // 是否发生了异常
            try
            {
                while (true)
                {
                    if (isBreakInfer == 1) break;
                    capture.Read(frame);//图像存储一帧数据
                    if (frame.Empty()) break;

                    // ------------- 原始图片 ----------
                    Bitmap image = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(frame); //显示原始图片到box1
                    if (pictureBox1.Image != null) pictureBox1.Image.Dispose();
                    pictureBox1.Image = image; //显示原始图片到box1
                    pictureBox1.SizeMode = PictureBoxSizeMode.Zoom; //显示原始图片到box1

                    // ------------- 送入推理的图片以及数据 ----------
                    image = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(frame);
                    byte[] inputData = GetBGRValues(image, out int stride);

                    float[] resultlist = new float[600];
                    IntPtr results = FloatToIntptr(resultlist);
                    int[] boxesInfo = new int[1];
                    byte[] labellist = new byte[1000];    //新建字节数组

                    TimeSpan infer_start_time = new TimeSpan(DateTime.Now.Ticks);
                    //第四个参数为输入图像的通道数
                    Det_ModelPredict(inputData, image.Width, image.Height, 3, results, boxesInfo, ref labellist[0]);
                    TimeSpan infer_end_time = new TimeSpan(DateTime.Now.Ticks);

                    string strGet = System.Text.Encoding.Default.GetString(labellist, 0, labellist.Length);    //将字节数组转换为字符串
                    string[] predict_Label_List = strGet.Split(' ');  // 预测的类别情况
                                                                      //Console.WriteLine("labellist: {0}", strGet);
                                                                      // 转换为mat数据，方便opencv处理
                    using OpenCvSharp.Mat mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(image);//用bitmap转换为mat
                    for (int i = 0; i < boxesInfo[0]; i++) // 未绘制图像
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

                            // 获取文本区域的大小
                            var text_size = Cv2.GetTextSize($"{predict_Label_List[i]}-{labelindex}-{score:f2}",
                                             HersheyFonts.HersheySimplex, 1, 2, out int baseline);  // 1倍大小的HersheySimplex，高度为22
                                                                                                    // 获取文本区域的左下顶点 -- 右上角
                            int left_down_x = (int)left + 22; // 小偏移调整量: (int)(text_size.Width/10)
                            int left_down_y = (int)top + text_size.Height;

                            // 绘制矩形
                            Cv2.Rectangle(mat, new OpenCvSharp.Rect((int)left, (int)top, (int)right, (int)down), new OpenCvSharp.Scalar(color_[0], color_[1], color_[2]), 2, LineTypes.AntiAlias);//LineTypes.AntiAlias:反锯齿效果
                            Cv2.PutText(mat, $"{predict_Label_List[i]}-{labelindex}-: {score:f2}", new OpenCvSharp.Point(left_down_x, left_down_y), HersheyFonts.HersheySimplex, 1, new OpenCvSharp.Scalar(color_[0], color_[1], color_[2]), 2, LineTypes.Link4);
                        }
                    }

                    // 转换回bitmap进行显示
                    image = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(mat);
                    // 反馈到另一个picturebox上
                    if (pictureBox2.Image != null) pictureBox2.Image.Dispose();
                    pictureBox2.Image = image;
                    pictureBox2.SizeMode = PictureBoxSizeMode.Zoom;

                    // 展示推理耗时
                    TimeSpan start2end_time = infer_end_time.Subtract(infer_start_time).Duration();
                    double cost_milliseconds = start2end_time.TotalMilliseconds;
                    // 通过委托展示到label上
                    Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };//定义一个委托
                    label7.Invoke(AsyncUIDelegate, new object[] { $"{cost_milliseconds:f2}" });
                }
            }
            catch (Exception e)
            {
                raise_ex_flag = 1;  // 发生了异常

                // 默认耗时为0ms
                Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };//定义一个委托
                label7.Invoke(AsyncUIDelegate, new object[] { "0.00" });
                MessageBox.Show("1.请检查模型文件与模型类型是否一致!\n2.内存溢出，yml预处理有误，图片格式确保为1/3通道...", "模型运行失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

            isBreakInfer = 0; // 清空标志

            // DestructModel(); // 销毁模型

            is_infer = 0; // 退出推理  -- 解除推理状态
            infer_video_img_flag = 0; // 重置当前推理状态 -- 解除图片推理状态
            if (raise_ex_flag == 0) MessageBox.Show("视频推理完成!"); // 未发生异常，正常显示推理完成提示
        }


        // 识别单张图片 -- 固定大小展示:short_side: 512
        private void cls_infer_one_img()
        {
            is_infer = 1; // 进入推理

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
            byte[] pre_category = new byte[200];    //新建字节数组

            int raise_ex_flag = 0;  // 是否发生了异常
            try
            {
                TimeSpan infer_start_time = new TimeSpan(DateTime.Now.Ticks);
                //第四个参数为输入图像的通道数
                Cls_ModelPredict(inputData, bmp.Width, bmp.Height, 3, ref pre_score[0], ref pre_category[0], ref pre_category_id[0]);
                TimeSpan infer_end_time = new TimeSpan(DateTime.Now.Ticks);

                string category_strGet = System.Text.Encoding.Default.GetString(pre_category, 0, pre_category.Length).Split('\0')[0];    //将类别字节数组转换为字符串
                OpenCvSharp.Mat mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(bmp);//用bitmap转换为mat

                // 对应类别的颜色
                int[] color_ = { (int)(color_map[(pre_category_id[0]%256)*3]),
                                        (int)(color_map[(pre_category_id[0] % 256) * 3 + 1]),
                                        (int)(color_map[(pre_category_id[0] % 256) * 3 + 2]) };
                // 获取文本区域的大小
                var text_size  = Cv2.GetTextSize($"{category_strGet}-{pre_category_id[0]}-{pre_score[0]:f2}",
                                 HersheyFonts.HersheySimplex, 1, 2, out int baseline);  // 1倍大小的HersheySimplex，高度为22
                // 获取文本区域的左下顶点 -- 右上角
                int left_down_x = bmp.Width - text_size.Width; // 小偏移调整量: (int)(text_size.Width/10)
                int left_down_y = text_size.Height;

                // 书写类别
                Cv2.PutText(mat, $"{category_strGet}-{pre_category_id[0]}-{pre_score[0]:f2}", new OpenCvSharp.Point(left_down_x, left_down_y), HersheyFonts.HersheySimplex, 1, new OpenCvSharp.Scalar(color_[0], color_[1], color_[2]), 2, LineTypes.Link4);
                
                // 转换回bitmap进行显示
                bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(mat);
                // 反馈到另一个picturebox上
                if (pictureBox2.Image != null) pictureBox2.Image.Dispose();
                pictureBox2.Image = bmp;
                pictureBox2.SizeMode = PictureBoxSizeMode.Zoom;

                // 收回内存
                input_mat = null;
                output_mat = null;
                mat = null;
                inputData = null;
                pre_score = null;
                pre_category_id = null;
                pre_category = null;

                // 展示推理耗时
                TimeSpan start2end_time = infer_end_time.Subtract(infer_start_time).Duration();
                double cost_milliseconds = start2end_time.TotalMilliseconds;
                // 通过委托展示到label上
                Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };//定义一个委托
                label7.Invoke(AsyncUIDelegate, new object[] { $"{cost_milliseconds:f2}" });

            }
            catch (Exception e)
            {
                raise_ex_flag = 1;

                // 默认耗时为0ms
                Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };//定义一个委托
                label7.Invoke(AsyncUIDelegate, new object[] { "0.00" });
                MessageBox.Show("1.请检查模型文件与模型类型是否一致!\n2.内存溢出，yml预处理有误，图片格式确保为1/3通道...", "模型运行失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

            isBreakInfer = 0; // 清空标志
            is_infer = 0; // 退出推理  -- 解除推理状态
            infer_one_img_flag = 0; // 重置当前推理状态 -- 解除图片推理状态
            if (raise_ex_flag == 0) MessageBox.Show("图片推理完成!"); // 未发生异常，正常显示推理完成提示
        }

        // 识别图片文件夹 -- 固定大小展示:short_side: 512
        private void cls_infer_many_img()
        {
            is_infer = 1; // 进入推理

            byte[] color_map = get_color_map_list(256);

            int raise_ex_flag = 0;  // 是否发生了异常
            try
            {
                foreach (string img_file in imgfiles)
                {
                    if (isBreakInfer == 1) break;  // 中断推理

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
                    byte[] pre_category = new byte[200];    //新建字节数组

                    TimeSpan infer_start_time = new TimeSpan(DateTime.Now.Ticks);
                    //第四个参数为输入图像的通道数
                    Cls_ModelPredict(inputData, bmp.Width, bmp.Height, 3, ref pre_score[0], ref pre_category[0], ref pre_category_id[0]);
                    TimeSpan infer_end_time = new TimeSpan(DateTime.Now.Ticks);

                    string category_strGet = System.Text.Encoding.Default.GetString(pre_category, 0, pre_category.Length).Split('\0')[0];    //将类别字节数组转换为字符串
                    OpenCvSharp.Mat mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(bmp);//用bitmap转换为mat

                    // 对应类别的颜色
                    int[] color_ = { (int)(color_map[(pre_category_id[0]%256)*3]),
                                            (int)(color_map[(pre_category_id[0] % 256) * 3 + 1]),
                                            (int)(color_map[(pre_category_id[0] % 256) * 3 + 2]) };
                    // 获取文本区域的大小
                    var text_size = Cv2.GetTextSize($"{category_strGet}-{pre_category_id[0]}-:{pre_score[0]:f2}",
                                     HersheyFonts.HersheySimplex, 1, 2, out int baseline);
                    // 获取文本区域的左下顶点 -- 右上角
                    int left_down_x = bmp.Width - text_size.Width; // 小偏移调整量: (int)(text_size.Width/10)
                    int left_down_y = text_size.Height;

                    // 书写类别
                    Cv2.PutText(mat, $"{category_strGet}-{pre_category_id[0]}-:{pre_score[0]:f2}", new OpenCvSharp.Point(left_down_x, left_down_y), HersheyFonts.HersheySimplex, 1, new OpenCvSharp.Scalar(color_[0], color_[1], color_[2]), 2, LineTypes.Link4);

                    // 转换回bitmap进行显示
                    bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(mat);

                    // 原图显示
                    Bitmap show_image = new Bitmap(img_file);
                    if (pictureBox1.Image != null) pictureBox1.Image.Dispose();
                    pictureBox1.Image = show_image; //显示原始图片到box1
                    pictureBox1.SizeMode = PictureBoxSizeMode.Zoom; //显示原始图片到box1

                    // 反馈到另一个picturebox上
                    if (pictureBox2.Image != null) pictureBox2.Image.Dispose();
                    pictureBox2.Image = bmp;
                    pictureBox2.SizeMode = PictureBoxSizeMode.Zoom;

                    // 收回内存
                    mat = null;
                    inputData = null;
                    pre_score = null;
                    pre_category_id = null;
                    pre_category = null;

                    // 展示推理耗时
                    TimeSpan start2end_time = infer_end_time.Subtract(infer_start_time).Duration();
                    double cost_milliseconds = start2end_time.TotalMilliseconds;
                    // 通过委托展示到label上
                    Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };//定义一个委托
                    label7.Invoke(AsyncUIDelegate, new object[] { $"{cost_milliseconds:f2}" });

                    Thread.Sleep(continue_infer_delay);  // 连续识别时，每张图片间隔continue_infer_delay毫秒
                }
            }
            catch (Exception e)
            {
                raise_ex_flag = 1;

                // 默认耗时为0ms
                Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };//定义一个委托
                label7.Invoke(AsyncUIDelegate, new object[] { "0.00" });
                MessageBox.Show("1.请检查模型文件与模型类型是否一致!\n2.内存溢出，yml预处理有误，图片格式确保为1/3通道...", "模型运行失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

            isBreakInfer = 0; // 清空标志

            is_infer = 0; // 退出推理  -- 解除推理状态
            infer_many_img_flag = 0;  // 重置当前推理状态 -- 解除图片推理状态
            if (raise_ex_flag == 0) MessageBox.Show("图片文件夹推理完成!"); // 未发生异常，正常显示推理完成提示
        }

        // 识别视频 -- 固定大小展示:short_side: 512
        private void cls_infer_video_img()
        {
            is_infer = 1; // 进入推理

            byte[] color_map = get_color_map_list(256);

            VideoCapture capture = new VideoCapture();
            capture.Open(videofile); // 读取视频

            using Mat frame = new Mat();

            int raise_ex_flag = 0;  // 是否发生了异常
            try
            {
                while (true)
                {
                    if (isBreakInfer == 1) break;
                    capture.Read(frame);//图像存储一帧数据
                    if (frame.Empty()) break;

                    // ------------- 原始图片 ----------
                    Bitmap image = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(frame); //显示原始图片到box1
                    if (pictureBox1.Image != null) pictureBox1.Image.Dispose();
                    pictureBox1.Image = image; //显示原始图片到box1
                    pictureBox1.SizeMode = PictureBoxSizeMode.Zoom; //显示原始图片到box1

                    // ------------- 送入推理的图片以及数据 ----------
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
                    byte[] pre_category = new byte[200];    //新建字节数组

                    TimeSpan infer_start_time = new TimeSpan(DateTime.Now.Ticks);
                    //第四个参数为输入图像的通道数
                    Cls_ModelPredict(inputData, image.Width, image.Height, 3, ref pre_score[0], ref pre_category[0], ref pre_category_id[0]);
                    TimeSpan infer_end_time = new TimeSpan(DateTime.Now.Ticks);

                    string category_strGet = System.Text.Encoding.Default.GetString(pre_category, 0, pre_category.Length).Split('\0')[0];    //将类别字节数组转换为字符串
                    OpenCvSharp.Mat mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(image);//用bitmap转换为mat

                    // 对应类别的颜色
                    int[] color_ = { (int)(color_map[(pre_category_id[0]%256)*3]),
                                        (int)(color_map[(pre_category_id[0] % 256) * 3 + 1]),
                                        (int)(color_map[(pre_category_id[0] % 256) * 3 + 2]) };
                    // 获取文本区域的大小
                    var text_size = Cv2.GetTextSize($"{category_strGet}-{pre_category_id[0]}-:{pre_score[0]:f2}",
                                     HersheyFonts.HersheySimplex, 1, 2, out int baseline);
                    // 获取文本区域的左下顶点 -- 右上角
                    int left_down_x = image.Width - text_size.Width; // 小偏移调整量: (int)(text_size.Width/10)
                    int left_down_y = text_size.Height;
                    // 书写类别
                    Cv2.PutText(mat, $"{category_strGet}-{pre_category_id[0]}-:{pre_score[0]:f2}", new OpenCvSharp.Point(left_down_x, left_down_y), HersheyFonts.HersheySimplex, 1, new OpenCvSharp.Scalar(color_[0], color_[1], color_[2]), 2, LineTypes.Link4);

                    // 转换回bitmap进行显示
                    image = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(mat);
                    // 反馈到另一个picturebox上
                    pictureBox2.Image = image;
                    pictureBox2.SizeMode = PictureBoxSizeMode.Zoom;

                    // 收回内存
                    mat = null;
                    inputData = null;
                    pre_score = null;
                    pre_category_id = null;
                    pre_category = null;

                    // 展示推理耗时
                    TimeSpan start2end_time = infer_end_time.Subtract(infer_start_time).Duration();
                    double cost_milliseconds = start2end_time.TotalMilliseconds;
                    // 通过委托展示到label上
                    Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };//定义一个委托
                    label7.Invoke(AsyncUIDelegate, new object[] { $"{cost_milliseconds:f2}" });
                }
            }
            catch (Exception e)
            {
                raise_ex_flag = 1;  // 发生了异常

                // 默认耗时为0ms
                Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };//定义一个委托
                label7.Invoke(AsyncUIDelegate, new object[] { "0.00" });
                MessageBox.Show("1.请检查模型文件与模型类型是否一致!\n2.内存溢出，yml预处理有误，图片格式确保为1/3通道...", "模型运行失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

            isBreakInfer = 0; // 清空标志

            // DestructModel(); // 销毁模型

            is_infer = 0; // 退出推理  -- 解除推理状态
            infer_video_img_flag = 0; // 重置当前推理状态 -- 解除图片推理状态
            if (raise_ex_flag == 0) MessageBox.Show("视频推理完成!"); // 未发生异常，正常显示推理完成提示
        }


        // 分割图片 -- 固定大小展示:512 X 512
        private void seg_infer_one_img()
        {
            is_infer = 1; // 进入推理

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

            byte[] output_map = new byte[input_bmp.Height * input_bmp.Width];    //新建字节数组

            int raise_ex_flag = 0;  // 是否发生了异常
            try
            {
                TimeSpan infer_start_time = new TimeSpan(DateTime.Now.Ticks);
                //第四个参数为输入图像的通道数
                Seg_ModelPredict(inputData, input_bmp.Width, input_bmp.Height, 3, ref output_map[0]);
                TimeSpan infer_end_time = new TimeSpan(DateTime.Now.Ticks);

                // 还原原始图像大小
                input_bmp = CreateBitmap(output_map, input_bmp.Width, input_bmp.Height, color_map); // 还原512的输入大小的图像

                input_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(input_bmp);
                output_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(input_bmp); // 获取处理后的图像
                Cv2.Resize(input_mat, output_mat, new OpenCvSharp.Size(origin_bmp.Width, origin_bmp.Height)); // 还原到与输入一致的图像大小
                input_mat = null;  // 回收内存

                input_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(origin_bmp); // 获取原始图像
                //OpenCvSharp.Mat add_mat = new Mat(); // 叠加后的图像

                Cv2.AddWeighted(output_mat, 1.0, input_mat, 0.35, 1, output_mat);  // 执行叠加
                input_bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(output_mat);

                // 反馈到另一个picturebox上
                if (pictureBox2.Image != null) pictureBox2.Image.Dispose();
                pictureBox2.Image = input_bmp;
                pictureBox2.SizeMode = PictureBoxSizeMode.Zoom;

                // 展示推理耗时
                TimeSpan start2end_time = infer_end_time.Subtract(infer_start_time).Duration();
                double cost_milliseconds = start2end_time.TotalMilliseconds;
                // 通过委托展示到label上
                Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };//定义一个委托
                label7.Invoke(AsyncUIDelegate, new object[] { $"{cost_milliseconds:f2}" });
            }
            catch (Exception e)
            {
                raise_ex_flag = 1;

                // 默认耗时为0ms
                Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };//定义一个委托
                label7.Invoke(AsyncUIDelegate, new object[] { "0.00" });
                MessageBox.Show("1.请检查模型文件与模型类型是否一致!\n2.内存溢出，yml预处理有误，图片格式确保为1/3通道...", "模型运行失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

            isBreakInfer = 0; // 清空标志
            is_infer = 0; // 退出推理  -- 解除推理状态
            infer_one_img_flag = 0; // 重置当前推理状态 -- 解除图片推理状态
            if (raise_ex_flag == 0) MessageBox.Show("图片推理完成!"); // 未发生异常，正常显示推理完成提示
        }

        // 分割图片文件夹 -- 固定大小展示:512 X 512
        private void seg_infer_many_img()
        {
            is_infer = 1; // 进入推理

            byte[] color_map = get_color_map_list(256);

            int raise_ex_flag = 0;  // 是否发生了异常
            try
            {
                foreach (string img_file in imgfiles)
                {
                    if (isBreakInfer == 1) break;  // 中断推理

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

                    byte[] output_map = new byte[input_bmp.Height * input_bmp.Width];    //新建字节数组

                    TimeSpan infer_start_time = new TimeSpan(DateTime.Now.Ticks);
                    //第四个参数为输入图像的通道数
                    Seg_ModelPredict(inputData, input_bmp.Width, input_bmp.Height, 3, ref output_map[0]);
                    TimeSpan infer_end_time = new TimeSpan(DateTime.Now.Ticks);

                    // 还原原始图像大小
                    input_bmp = CreateBitmap(output_map, input_bmp.Width, input_bmp.Height, color_map); // 还原512的输入大小的图像

                    input_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(input_bmp);
                    output_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(input_bmp); // 获取处理后的图像
                    Cv2.Resize(input_mat, output_mat, new OpenCvSharp.Size(origin_bmp.Width, origin_bmp.Height)); // 还原到与输入一致的图像大小
                    input_mat = null;  // 回收内存

                    input_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(origin_bmp); // 获取原始图像
                                                                                          //OpenCvSharp.Mat add_mat = new Mat(); // 叠加后的图像

                    Cv2.AddWeighted(output_mat, 1.0, input_mat, 0.35, 1, output_mat);  // 执行叠加
                    input_bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(output_mat);

                    // 显示图片
                    if (pictureBox1.Image != null) pictureBox1.Image.Dispose();
                    pictureBox1.Image = origin_bmp; //显示原始图片到box1
                    pictureBox1.SizeMode = PictureBoxSizeMode.Zoom; //显示原始图片到box1

                    if (pictureBox2.Image != null) pictureBox2.Image.Dispose();
                    pictureBox2.Image = input_bmp;
                    pictureBox2.SizeMode = PictureBoxSizeMode.Zoom;

                    // 展示推理耗时
                    TimeSpan start2end_time = infer_end_time.Subtract(infer_start_time).Duration();
                    double cost_milliseconds = start2end_time.TotalMilliseconds;
                    // 通过委托展示到label上
                    Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };//定义一个委托
                    label7.Invoke(AsyncUIDelegate, new object[] { $"{cost_milliseconds:f2}" });

                    Thread.Sleep(continue_infer_delay);  // 连续识别时，每张图片间隔continue_infer_delay毫秒
                }
            }
            catch (Exception e)
            {
                raise_ex_flag = 1;

                // 默认耗时为0ms
                Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };//定义一个委托
                label7.Invoke(AsyncUIDelegate, new object[] { "0.00" });
                MessageBox.Show("1.请检查模型文件与模型类型是否一致!\n2.内存溢出，yml预处理有误，图片格式确保为1/3通道...", "模型运行失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

            isBreakInfer = 0; // 清空标志

            is_infer = 0; // 退出推理  -- 解除推理状态
            infer_many_img_flag = 0;  // 重置当前推理状态 -- 解除图片推理状态
            if (raise_ex_flag == 0) MessageBox.Show("图片文件夹推理完成!"); // 未发生异常，正常显示推理完成提示
        }

        // 分割视频流 -- 固定大小展示:512 X 512
        private void seg_infer_video_img()
        {
            is_infer = 1; // 进入推理

            byte[] color_map = get_color_map_list(256);

            VideoCapture capture = new VideoCapture();
            capture.Open(videofile); // 读取视频

            using Mat frame = new Mat();

            int raise_ex_flag = 0;  // 是否发生了异常
            try
            {
                while (true)
                {
                    if (isBreakInfer == 1) break;
                    capture.Read(frame);//图像存储一帧数据
                    if (frame.Empty()) break;

                    // ------------- 原始图片 ----------
                    Bitmap origin_bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(frame);
                    // ------------- 送入推理的图片以及数据 ----------
                    Bitmap input_bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(frame);
                    // resize()
                    OpenCvSharp.Mat input_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(origin_bmp);
                    OpenCvSharp.Mat output_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(origin_bmp);

                    Cv2.Resize(input_mat, output_mat, new OpenCvSharp.Size(512, 512));
                    input_bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(output_mat);
                    input_mat = null;
                    output_mat = null;

                    byte[] inputData = GetBGRValues(input_bmp, out int stride);

                    byte[] output_map = new byte[input_bmp.Height * input_bmp.Width];    //新建字节数组

                    TimeSpan infer_start_time = new TimeSpan(DateTime.Now.Ticks);
                    //第四个参数为输入图像的通道数
                    Seg_ModelPredict(inputData, input_bmp.Width, input_bmp.Height, 3, ref output_map[0]);
                    TimeSpan infer_end_time = new TimeSpan(DateTime.Now.Ticks);

                    // 还原原始图像大小
                    input_bmp = CreateBitmap(output_map, input_bmp.Width, input_bmp.Height, color_map); // 还原512的输入大小的图像

                    input_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(input_bmp);
                    output_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(input_bmp); // 获取处理后的图像
                    Cv2.Resize(input_mat, output_mat, new OpenCvSharp.Size(origin_bmp.Width, origin_bmp.Height)); // 还原到与输入一致的图像大小
                    input_mat = null;  // 回收内存

                    input_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(origin_bmp); // 获取原始图像
                                                                                          //OpenCvSharp.Mat add_mat = new Mat(); // 叠加后的图像

                    Cv2.AddWeighted(output_mat, 1.0, input_mat, 0.35, 1, output_mat);  // 执行叠加
                    input_bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(output_mat);
                    

                    // 显示图片
                    if (pictureBox1.Image != null) pictureBox1.Image.Dispose();
                    pictureBox1.Image = origin_bmp; //显示原始图片到box1
                    pictureBox1.SizeMode = PictureBoxSizeMode.Zoom; //显示原始图片到box1

                    if (pictureBox2.Image != null) pictureBox2.Image.Dispose();
                    pictureBox2.Image = input_bmp;
                    pictureBox2.SizeMode = PictureBoxSizeMode.Zoom;

                    // 展示推理耗时
                    TimeSpan start2end_time = infer_end_time.Subtract(infer_start_time).Duration();
                    double cost_milliseconds = start2end_time.TotalMilliseconds;
                    // 通过委托展示到label上
                    Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };//定义一个委托
                    label7.Invoke(AsyncUIDelegate, new object[] { $"{cost_milliseconds:f2}" });
                }
            }
            catch (Exception e)
            {
                raise_ex_flag = 1;  // 发生了异常

                // 默认耗时为0ms
                Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };//定义一个委托
                label7.Invoke(AsyncUIDelegate, new object[] { "0.00" });
                MessageBox.Show("1.请检查模型文件与模型类型是否一致!\n2.内存溢出，yml预处理有误，图片格式确保为1/3通道...", "模型运行失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

            isBreakInfer = 0; // 清空标志

            // DestructModel(); // 销毁模型

            is_infer = 0; // 退出推理  -- 解除推理状态
            infer_video_img_flag = 0; // 重置当前推理状态 -- 解除图片推理状态
            if (raise_ex_flag == 0) MessageBox.Show("视频推理完成!"); // 未发生异常，正常显示推理完成提示
        }


        // MaskRCNN检测单张图片 -- GPU推理正常
        private void mask_infer_one_img()
        {
            is_infer = 1; // 进入推理

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

            byte[] mask_results = new byte[input_bmp.Height * input_bmp.Width];    //新建字节数组

            int[] boxesInfo = new int[1]; // 10 boundingbox
            byte[] labellist = new byte[1000];    //新建字节数组：label1_str label2_str 

            int raise_ex_flag = 0;  // 是否发生了异常
            try
            {
                TimeSpan infer_start_time = new TimeSpan(DateTime.Now.Ticks);
                Mask_ModelPredict(inputData, input_bmp.Width, input_bmp.Height, 3, results, ref mask_results[0], boxesInfo, ref labellist[0]);
                TimeSpan infer_end_time = new TimeSpan(DateTime.Now.Ticks);

                input_bmp = CreateBitmap(mask_results, input_bmp.Width, input_bmp.Height, color_map);

                output_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(input_bmp); // 获取处理后的图像
                input_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(origin_bmp); // 获取原始图像
                //OpenCvSharp.Mat add_mat = new Mat(); // 叠加后的图像

                Cv2.AddWeighted(output_mat, 0.65, input_mat, 0.35, 1, output_mat);  // 执行叠加
                input_bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(output_mat);

                string strGet = System.Text.Encoding.Default.GetString(labellist, 0, labellist.Length);    //将字节数组转换为字符串
                string[] predict_Label_List = strGet.Split(' ');  // 预测的类别情况
                                                                  // MessageBox.Show($"Box_Number: {boxesInfo[0]}");
                using OpenCvSharp.Mat mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(input_bmp);//用bitmap转换为mat
                for (int i = 0; i < boxesInfo[0]; i++) // 未绘制图像
                {
                    int labelindex = Convert.ToInt32(resultlist[i * 6 + 0]);
                    float score = resultlist[i * 6 + 1];
                    float left = resultlist[i * 6 + 2];
                    float top = resultlist[i * 6 + 3];
                    float right = resultlist[i * 6 + 4];
                    float down = resultlist[i * 6 + 5];

                    if (score > det_threshold)
                    {
                        labelindex += 1; // Mask RCNN包含背景，故而加1
                        int[] color_ = { (int)(color_map[(labelindex%256)*3]),
                                             (int)(color_map[(labelindex % 256) * 3 + 1]),
                                             (int)(color_map[(labelindex % 256) * 3 + 2]) };

                        labelindex -= 1; // 还原类别
                        // 获取文本区域的大小
                        var text_size = Cv2.GetTextSize($"{predict_Label_List[i]}-{labelindex}-{score:f2}",
                                         HersheyFonts.HersheySimplex, 1, 2, out int baseline);  // 1倍大小的HersheySimplex，高度为22
                                                                                                // 获取文本区域的左下顶点 -- 右上角
                        int left_down_x = (int)left + 22; // 小偏移调整量: (int)(text_size.Width/10)
                        int left_down_y = (int)top + text_size.Height;

                        // 绘制矩形，书写类别
                        Cv2.Rectangle(mat, new OpenCvSharp.Rect((int)left, (int)top, (int)right, (int)down), new OpenCvSharp.Scalar(color_[0], color_[1], color_[2]), 2, LineTypes.AntiAlias);//LineTypes.AntiAlias:反锯齿效果
                        Cv2.PutText(mat, $"{predict_Label_List[i]}-{labelindex}-: {score:f2}", new OpenCvSharp.Point(left_down_x, left_down_y), HersheyFonts.HersheySimplex, 1, new OpenCvSharp.Scalar(color_[0], color_[1], color_[2]), 2, LineTypes.Link4);
                    }
                }

                // 转换回bitmap进行显示
                input_bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(mat);
                if (pictureBox2.Image != null) pictureBox2.Image.Dispose();
                pictureBox2.Image = input_bmp;
                pictureBox2.SizeMode = PictureBoxSizeMode.Zoom;

                // 展示推理耗时
                TimeSpan start2end_time = infer_end_time.Subtract(infer_start_time).Duration();
                double cost_milliseconds = start2end_time.TotalMilliseconds;
                // 通过委托展示到label上
                Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };//定义一个委托
                label7.Invoke(AsyncUIDelegate, new object[] { $"{cost_milliseconds:f2}" });

            }
            catch (Exception e)
            {
                raise_ex_flag = 1;

                // 默认耗时为0ms
                Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };//定义一个委托
                label7.Invoke(AsyncUIDelegate, new object[] { "0.00" });
                MessageBox.Show("1.请检查模型文件与模型类型是否一致!\n2.内存溢出，yml预处理有误，图片格式确保为1/3通道...", "模型运行失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

            isBreakInfer = 0; // 清空标志
            is_infer = 0; // 退出推理  -- 解除推理状态
            infer_one_img_flag = 0; // 重置当前推理状态 -- 解除图片推理状态
            if (raise_ex_flag == 0) MessageBox.Show("图片推理完成!"); // 未发生异常，正常显示推理完成提示
        }

        // MaskRCNN检测图片文件夹
        private void mask_infer_many_img()
        {
            is_infer = 1; // 进入推理

            byte[] color_map = get_color_map_list(256);

            int raise_ex_flag = 0;  // 是否发生了异常
            try
            {
                foreach (string img_file in imgfiles)
                {
                    if (isBreakInfer == 1) break;  // 中断推理

                    Bitmap origin_bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(Cv2.ImRead(img_file));
                    Bitmap input_bmp = null;

                    // resize()
                    OpenCvSharp.Mat input_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(origin_bmp);
                    OpenCvSharp.Mat output_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(origin_bmp);

                    input_bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(output_mat);
                    byte[] inputData = GetBGRValues(origin_bmp, out int stride);

                    float[] resultlist = new float[600];
                    IntPtr results = FloatToIntptr(resultlist);

                    byte[] mask_results = new byte[input_bmp.Height * input_bmp.Width];    //新建字节数组

                    int[] boxesInfo = new int[1]; // 10 boundingbox
                    byte[] labellist = new byte[1000];    //新建字节数组：label1_str label2_str 

                    TimeSpan infer_start_time = new TimeSpan(DateTime.Now.Ticks);
                    //第四个参数为输入图像的通道数
                    Mask_ModelPredict(inputData, input_bmp.Width, input_bmp.Height, 3, results, ref mask_results[0], boxesInfo, ref labellist[0]);
                    TimeSpan infer_end_time = new TimeSpan(DateTime.Now.Ticks);

                    input_bmp = CreateBitmap(mask_results, input_bmp.Width, input_bmp.Height, color_map);

                    output_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(input_bmp); // 获取处理后的图像
                    input_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(origin_bmp); // 获取原始图像
                                                                                          //OpenCvSharp.Mat add_mat = new Mat(); // 叠加后的图像

                    Cv2.AddWeighted(output_mat, 1.0, input_mat, 0.35, 1, output_mat);  // 执行叠加
                    input_bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(output_mat);

                    string strGet = System.Text.Encoding.Default.GetString(labellist, 0, labellist.Length);    //将字节数组转换为字符串
                    string[] predict_Label_List = strGet.Split(' ');  // 预测的类别情况
                                                                      // MessageBox.Show($"Box_Number: {boxesInfo[0]}");
                    using OpenCvSharp.Mat mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(input_bmp);//用bitmap转换为mat
                    for (int i = 0; i < boxesInfo[0]; i++) // 未绘制图像
                    {
                        int labelindex = Convert.ToInt32(resultlist[i * 6 + 0]);
                        float score = resultlist[i * 6 + 1];
                        float left = resultlist[i * 6 + 2];
                        float top = resultlist[i * 6 + 3];
                        float right = resultlist[i * 6 + 4];
                        float down = resultlist[i * 6 + 5];

                        if (score > det_threshold)
                        {
                            labelindex += 1; // Mask RCNN包含背景，故而加1
                            int[] color_ = { (int)(color_map[(labelindex%256)*3]),
                                             (int)(color_map[(labelindex % 256) * 3 + 1]),
                                             (int)(color_map[(labelindex % 256) * 3 + 2]) };

                            labelindex -= 1; // 还原类别
                                             // 获取文本区域的大小
                            var text_size = Cv2.GetTextSize($"{predict_Label_List[i]}-{labelindex}-{score:f2}",
                                             HersheyFonts.HersheySimplex, 1, 2, out int baseline);  // 1倍大小的HersheySimplex，高度为22
                                                                                                    // 获取文本区域的左下顶点 -- 右上角
                            int left_down_x = (int)left + 22; // 小偏移调整量: (int)(text_size.Width/10)
                            int left_down_y = (int)top + text_size.Height;

                            // 绘制矩形，书写类别
                            Cv2.Rectangle(mat, new OpenCvSharp.Rect((int)left, (int)top, (int)right, (int)down), new OpenCvSharp.Scalar(color_[0], color_[1], color_[2]), 2, LineTypes.AntiAlias);//LineTypes.AntiAlias:反锯齿效果
                            Cv2.PutText(mat, $"{predict_Label_List[i]}-{labelindex}-: {score:f2}", new OpenCvSharp.Point(left_down_x, left_down_y), HersheyFonts.HersheySimplex, 1, new OpenCvSharp.Scalar(color_[0], color_[1], color_[2]), 2, LineTypes.Link4);
                        }
                    }

                    // 显示图片
                    if (pictureBox1.Image != null) pictureBox1.Image.Dispose();
                    pictureBox1.Image = origin_bmp; //显示原始图片到box1
                    pictureBox1.SizeMode = PictureBoxSizeMode.Zoom; //显示原始图片到box1

                    // 转换回bitmap进行显示
                    input_bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(mat);
                    if (pictureBox2.Image != null) pictureBox2.Image.Dispose();
                    pictureBox2.Image = input_bmp;
                    pictureBox2.SizeMode = PictureBoxSizeMode.Zoom;

                    // 展示推理耗时
                    TimeSpan start2end_time = infer_end_time.Subtract(infer_start_time).Duration();
                    double cost_milliseconds = start2end_time.TotalMilliseconds;
                    // 通过委托展示到label上
                    Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };//定义一个委托
                    label7.Invoke(AsyncUIDelegate, new object[] { $"{cost_milliseconds:f2}" });

                    Thread.Sleep(continue_infer_delay);  // 连续识别时，每张图片间隔continue_infer_delay毫秒
                }
            }
            catch (Exception e)
            {
                raise_ex_flag = 1;

                // 默认耗时为0ms
                Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };//定义一个委托
                label7.Invoke(AsyncUIDelegate, new object[] { "0.00" });
                MessageBox.Show("1.请检查模型文件与模型类型是否一致!\n2.内存溢出，yml预处理有误，图片格式确保为1/3通道...", "模型运行失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

            isBreakInfer = 0; // 清空标志

            is_infer = 0; // 退出推理  -- 解除推理状态
            infer_many_img_flag = 0;  // 重置当前推理状态 -- 解除图片推理状态
            if (raise_ex_flag == 0) MessageBox.Show("图片文件夹推理完成!"); // 未发生异常，正常显示推理完成提示
        }

        // MaskRCNN检测视频流
        private void mask_infer_video_img()
        {
            is_infer = 1; // 进入推理

            byte[] color_map = get_color_map_list(256);

            VideoCapture capture = new VideoCapture();
            capture.Open(videofile); // 读取视频

            using Mat frame = new Mat();

            int raise_ex_flag = 0;  // 是否发生了异常
            try
            {
                while (true)
                {
                    if (isBreakInfer == 1) break;
                    capture.Read(frame);//图像存储一帧数据
                    if (frame.Empty()) break;

                    // ------------- 原始图片 ----------
                    Bitmap origin_bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(frame);
                    // ------------- 送入推理的图片以及数据 ----------
                    Bitmap input_bmp = null;

                    // resize()
                    OpenCvSharp.Mat input_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(origin_bmp);
                    OpenCvSharp.Mat output_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(origin_bmp);

                    input_bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(output_mat);
                    byte[] inputData = GetBGRValues(origin_bmp, out int stride);

                    float[] resultlist = new float[600];
                    IntPtr results = FloatToIntptr(resultlist);

                    byte[] mask_results = new byte[input_bmp.Height * input_bmp.Width];    //新建字节数组

                    int[] boxesInfo = new int[1]; // 10 boundingbox
                    byte[] labellist = new byte[1000];    //新建字节数组：label1_str label2_str 

                    TimeSpan infer_start_time = new TimeSpan(DateTime.Now.Ticks);
                    //第四个参数为输入图像的通道数
                    Mask_ModelPredict(inputData, input_bmp.Width, input_bmp.Height, 3, results, ref mask_results[0], boxesInfo, ref labellist[0]);
                    TimeSpan infer_end_time = new TimeSpan(DateTime.Now.Ticks);

                    input_bmp = CreateBitmap(mask_results, input_bmp.Width, input_bmp.Height, color_map);

                    output_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(input_bmp); // 获取处理后的图像
                    input_mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(origin_bmp); // 获取原始图像
                                                                                          //OpenCvSharp.Mat add_mat = new Mat(); // 叠加后的图像

                    Cv2.AddWeighted(output_mat, 1.0, input_mat, 0.35, 1, output_mat);  // 执行叠加
                    input_bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(output_mat);

                    string strGet = System.Text.Encoding.Default.GetString(labellist, 0, labellist.Length);    //将字节数组转换为字符串
                    string[] predict_Label_List = strGet.Split(' ');  // 预测的类别情况
                                                                      // MessageBox.Show($"Box_Number: {boxesInfo[0]}");
                    using OpenCvSharp.Mat mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(input_bmp);//用bitmap转换为mat
                    for (int i = 0; i < boxesInfo[0]; i++) // 未绘制图像
                    {
                        int labelindex = Convert.ToInt32(resultlist[i * 6 + 0]);
                        float score = resultlist[i * 6 + 1];
                        float left = resultlist[i * 6 + 2];
                        float top = resultlist[i * 6 + 3];
                        float right = resultlist[i * 6 + 4];
                        float down = resultlist[i * 6 + 5];

                        if (score > det_threshold)
                        {
                            labelindex += 1; // Mask RCNN包含背景，故而加1
                            int[] color_ = { (int)(color_map[(labelindex%256)*3]),
                                             (int)(color_map[(labelindex % 256) * 3 + 1]),
                                             (int)(color_map[(labelindex % 256) * 3 + 2]) };

                            labelindex -= 1; // 还原类别
                                             // 获取文本区域的大小
                            var text_size = Cv2.GetTextSize($"{predict_Label_List[i]}-{labelindex}-{score:f2}",
                                             HersheyFonts.HersheySimplex, 1, 2, out int baseline);  // 1倍大小的HersheySimplex，高度为22
                                                                                                    // 获取文本区域的左下顶点 -- 右上角
                            int left_down_x = (int)left + 22; // 小偏移调整量: (int)(text_size.Width/10)
                            int left_down_y = (int)top + text_size.Height;

                            // 绘制矩形，书写类别
                            Cv2.Rectangle(mat, new OpenCvSharp.Rect((int)left, (int)top, (int)right, (int)down), new OpenCvSharp.Scalar(color_[0], color_[1], color_[2]), 2, LineTypes.AntiAlias);//LineTypes.AntiAlias:反锯齿效果
                            Cv2.PutText(mat, $"{predict_Label_List[i]}-{labelindex}-: {score:f2}", new OpenCvSharp.Point(left_down_x, left_down_y), HersheyFonts.HersheySimplex, 1, new OpenCvSharp.Scalar(color_[0], color_[1], color_[2]), 2, LineTypes.Link4);
                        }
                    }

                    // 显示图片
                    if (pictureBox1.Image != null) pictureBox1.Image.Dispose();
                    pictureBox1.Image = origin_bmp; //显示原始图片到box1
                    pictureBox1.SizeMode = PictureBoxSizeMode.Zoom; //显示原始图片到box1

                    // 转换回bitmap进行显示
                    input_bmp = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(mat);
                    if (pictureBox2.Image != null) pictureBox2.Image.Dispose();
                    pictureBox2.Image = input_bmp;
                    pictureBox2.SizeMode = PictureBoxSizeMode.Zoom;

                    // 展示推理耗时
                    TimeSpan start2end_time = infer_end_time.Subtract(infer_start_time).Duration();
                    double cost_milliseconds = start2end_time.TotalMilliseconds;
                    // 通过委托展示到label上
                    Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };//定义一个委托
                    label7.Invoke(AsyncUIDelegate, new object[] { $"{cost_milliseconds:f2}" });
                }
            }
            catch (Exception e)
            {
                raise_ex_flag = 1;  // 发生了异常

                // 默认耗时为0ms
                Action<String> AsyncUIDelegate = delegate (string n) { label7.Text = n; };//定义一个委托
                label7.Invoke(AsyncUIDelegate, new object[] { "0.00" });
                MessageBox.Show("1.请检查模型文件与模型类型是否一致!\n2.内存溢出，yml预处理有误，图片格式确保为1/3通道...", "模型运行失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

            isBreakInfer = 0; // 清空标志

            is_infer = 0; // 退出推理  -- 解除推理状态
            infer_video_img_flag = 0; // 重置当前推理状态 -- 解除图片推理状态
            if (raise_ex_flag == 0) MessageBox.Show("视频推理完成!"); // 未发生异常，正常显示推理完成提示
        }

        /**********************************************************************/
        /*****************          7.部分推理组件函数          ***************/
        /**********************************************************************/
        ///   <summary>
        ///  从内存流中指定位置，读取数据
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
        ///  使用byte[]数据，生成三通道 BMP 位图
        ///   </summary>
        ///   <param name="originalImageData"></param>
        ///   <param name="originalWidth"></param>
        ///   <param name="originalHeight"></param>
        ///   <returns></returns>
        public static Bitmap CreateBitmap(byte[] originalImageData, int originalWidth, int originalHeight, byte[] color_map)
        {
            // 指定8位格式，即256色
            Bitmap resultBitmap = new Bitmap(originalWidth, originalHeight, System.Drawing.Imaging.PixelFormat.Format8bppIndexed);

            // 将该位图存入内存中
            MemoryStream curImageStream = new MemoryStream();
            resultBitmap.Save(curImageStream, System.Drawing.Imaging.ImageFormat.Bmp);
            curImageStream.Flush();

            // 由于位图数据需要DWORD对齐（4byte倍数），计算需要补位的个数
            int curPadNum = ((originalWidth * 8 + 31) / 32 * 4) - originalWidth;

            // 最终生成的位图数据大小
            int bitmapDataSize = ((originalWidth * 8 + 31) / 32 * 4) * originalHeight;

            // 数据部分相对文件开始偏移，具体可以参考位图文件格式
            int dataOffset = ReadData(curImageStream, 10, 4);


            // 改变调色板，因为默认的调色板是32位彩色的，需要修改为256色的调色板
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

            // 最终生成的位图数据，以及大小，高度没有变，宽度需要调整
            byte[] destImageData = new byte[bitmapDataSize];
            int destWidth = originalWidth + curPadNum;

            // 生成最终的位图数据，注意的是，位图数据 从左到右，从下到上，所以需要颠倒
            for (int originalRowIndex = originalHeight - 1; originalRowIndex >= 0; originalRowIndex--)
            {
                int destRowIndex = originalHeight - originalRowIndex - 1;

                for (int dataIndex = 0; dataIndex < originalWidth; dataIndex++)
                {
                    // 同时还要注意，新的位图数据的宽度已经变化destWidth，否则会产生错位
                    destImageData[destRowIndex * destWidth + dataIndex] = originalImageData[originalRowIndex * originalWidth + dataIndex];
                }
            }

            // 将流的Position移到数据段   
            curImageStream.Position = dataOffset;

            // 将新位图数据写入内存中
            curImageStream.Write(destImageData, 0, bitmapDataSize);

            curImageStream.Flush();

            // 将内存中的位图写入Bitmap对象
            resultBitmap = new Bitmap(curImageStream);

            resultBitmap = transForm8to24(resultBitmap, color_map);  // 转为3通道图像

            return resultBitmap;
        }

        // 实现bitmap单通道到三通道(分割生成掩码图像(单通道) ==> RGB图像)
        public static Bitmap transForm8to24(Bitmap bmp, byte[] color_map)
        {

            System.Drawing.Rectangle rect = new System.Drawing.Rectangle(0, 0, bmp.Width, bmp.Height);

            System.Drawing.Imaging.BitmapData bitmapData = bmp.LockBits(rect, System.Drawing.Imaging.ImageLockMode.ReadOnly, bmp.PixelFormat);

            //计算实际8位图容量
            int size8 = bitmapData.Stride * bmp.Height;
            byte[] grayValues = new byte[size8];

            //// 申请目标位图的变量，并将其内存区域锁定  
            Bitmap TempBmp = new Bitmap(bmp.Width, bmp.Height, PixelFormat.Format24bppRgb);
            BitmapData TempBmpData = TempBmp.LockBits(new Rectangle(0, 0, bmp.Width, bmp.Height), ImageLockMode.WriteOnly, PixelFormat.Format24bppRgb);


            //// 获取图像参数以及设置24位图信息 
            int stride = TempBmpData.Stride;  // 扫描线的宽度  
            int offset = stride - TempBmp.Width;  // 显示宽度与扫描线宽度的间隙  
            IntPtr iptr = TempBmpData.Scan0;  // 获取bmpData的内存起始位置  
            int scanBytes = stride * TempBmp.Height;// 用stride宽度，表示这是内存区域的大小  

            //// 下面把原始的显示大小字节数组转换为内存中实际存放的字节数组  

            byte[] pixelValues = new byte[scanBytes];  //为目标数组分配内存  
            System.Runtime.InteropServices.Marshal.Copy(bitmapData.Scan0, grayValues, 0, size8);
            
            for (int i = 0; i < bmp.Height; i++)
            {

                for (int j = 0; j < bitmapData.Stride; j++)
                {

                    if (j >= bmp.Width)
                        continue;


                    int indexSrc = i * bitmapData.Stride + j;
                    int realIndex = i * TempBmpData.Stride + j * 3;

                    // color_id：就是预测出来的结果
                    int color_id = (int)grayValues[indexSrc] % 256;

                    if (color_id == 0) // 分割中类别1对应值1，而背景往往为0，因此这里就将背景置为[0, 0, 0]
                    {
                        // 空白
                        pixelValues[realIndex] = 0;
                        pixelValues[realIndex + 1] = 0;
                        pixelValues[realIndex + 2] = 0;
                    }
                    else
                    {
                        // 替换为color_map中的颜色值
                        pixelValues[realIndex] = color_map[color_id * 3];
                        pixelValues[realIndex + 1] = color_map[color_id * 3 + 1];
                        pixelValues[realIndex + 2] = color_map[color_id * 3 + 2];
                    }

                }

            }

            //// 用Marshal的Copy方法，将刚才得到的内存字节数组复制到BitmapData中  
            System.Runtime.InteropServices.Marshal.Copy(pixelValues, 0, iptr, scanBytes);
            TempBmp.UnlockBits(TempBmpData);  // 解锁内存区域  
            bmp.UnlockBits(bitmapData);

            return TempBmp;
        }

        // 生成伪彩色图的RGB值集合(color_map) -- 同时也是适用于检测框分类颜色
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

            // 去掉底色
            color_map = color_map.Skip(3).ToArray();

            return color_map;
        }

        /// <summary>
        /// 获得目录下所有文件或指定文件类型文件(包含所有子文件夹)
        /// </summary>
        /// <param name="path">文件夹路径</param>
        /// <param name="extName">扩展名可以多个 例如 .mp3.wma.rm</param>
        /// <returns>List<FileInfo></returns>
        public static List<FileInfo> getFile(string path, string extName, List<FileInfo> lst)
        {
            try
            {
                DirectoryInfo fdir = new DirectoryInfo(path);
                FileInfo[] file = fdir.GetFiles();
                //FileInfo[] file = Directory.GetFiles(path); //文件列表
                if (file.Length != 0) //当前目录文件或文件夹不为空
                {
                    foreach (FileInfo f in file) //显示当前目录所有文件
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

        // 创建指向float数组类型的IntPtr指针
        public static IntPtr FloatToIntptr(float[] bytes)
        {
            GCHandle hObject = GCHandle.Alloc(bytes, GCHandleType.Pinned);
            return hObject.AddrOfPinnedObject();
        }

        // 检查MaskRCNN模型是否启动在GPU上 -- 只支持GPU推理，因为内存占用较大，CPU可能溢出，导致无法连续推理
        public static bool CheckMaskRCNN_workOnGpu(string model_type, bool use_gpu)
        {
            if (model_type == "mask")
            {
                if (use_gpu == false) // 当且仅当为MaskRCNN时，没有使用GPU会返回false
                    return false;
            }
            return true;
        }
    }
}
