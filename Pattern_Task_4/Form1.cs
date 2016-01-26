using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Pattern_Task_4
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
            Class1 obj = new Class1();
            obj.readfile();
            obj.calculateMeans();
            obj.calculateCovariances();
            obj.classifier();

            //for (int i = 0; i < obj.Confusion.ColumnCount; i++)
            //{
            //    dataGridView1.Columns[i].Name = "C" + (i).ToString();
            //}

           
            //for (int i = 0; i < obj.Confusion.RowCount; i++)
            //{
            //    dataGridView1.Rows.Add();
                
            //}
            label1.Text = "Accuracy = "+obj.percentage.ToString()+" % ";
        }
    }
}
