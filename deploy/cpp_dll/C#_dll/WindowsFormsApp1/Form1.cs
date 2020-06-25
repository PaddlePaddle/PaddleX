using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Runtime.InteropServices;
namespace WindowsFormsApp1
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }
        [DllImport("detector.dll", EntryPoint = "Loadmodel", CharSet = CharSet.Ansi)]
        public static extern void test();
        private void button1_Click(object sender, EventArgs e)
        {
            test();
        }
    }
}
