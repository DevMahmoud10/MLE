using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;


namespace Pattern_Task_4
{
    class Class1
    {
        List<Vector<double>> evidence = new List<Vector<double>>();
        Matrix<double> Data = DenseMatrix.OfArray(new double[150, 4]);
        List<Vector<double>> means = new List<Vector<double>>();
        List<Matrix<double>> covariances = new List<Matrix<double>>();
        List<Matrix<double>> likelihood = new List<Matrix<double>>();
        List<Matrix<double>> Posterior = new List<Matrix<double>>();
        public Matrix<double> Confusion = DenseMatrix.OfArray(new double[3, 3]);
        public double percentage;

        public Class1()
        {
            for (int i = 0; i < 3; i++)
            {
                means.Add(Vector<double>.Build.Dense(4));

                covariances.Add(Matrix<double>.Build.Dense(4, 4));
                likelihood.Add(Matrix<double>.Build.Dense(30, 3));
                Posterior.Add(Matrix<double>.Build.Dense(30, 3));
                evidence.Add(Vector<double>.Build.Dense(30));
            }


        }
        public void readfile()
        {
            FileStream fs = new FileStream("Iris Data.txt", FileMode.Open, FileAccess.Read);
            var streamReader = new StreamReader(fs);
            string line = "";

            int i = 0; // number of samples
            while ((line = streamReader.ReadLine()) != null)
            {

                string[] tmp = new string[3];
                //data[i].Add(Int32.Parse(line.Split(',')));
                tmp = line.Split(',');
                for (int j = 0; j < 4; j++)
                {

                    this.Data[i, j] = double.Parse(tmp[j]);
                }
                i++;
            }
        }

        public void calculateMeans()
        {
            for (int i = 0; i < 3; i++)
            {
                switch (i)
                {
                    case 0:
                        {
                            for (int j = 0; j < 4; j++)
                            {
                                double sum = 0;
                                for (int k = 0; k < 20; k++)
                                {
                                    sum += Data[k, j];
                                }
                                means[i][j] = sum / 20;
                            }
                        }
                        break;

                    case 1:
                        {
                            for (int j = 0; j < 4; j++)
                            {
                                double sum = 0;
                                for (int k = 50; k < 70; k++)
                                {
                                    sum += Data[k, j];
                                }
                                means[i][j] = sum / 20;
                            }
                        }
                        break;


                    case 2:
                        {
                            for (int j = 0; j < 4; j++)
                            {
                                double sum = 0;
                                for (int k = 100; k < 120; k++)
                                {
                                    sum += Data[k, j];
                                }
                                means[i][j] = sum / 20;
                            }
                        }
                        break;
                }

            }


        }

        public void calculateCovariances()
        {
            for (int l = 0; l < 3; l++) //list of classes
            {
                for (int i = 0; i < 4; i++)//rows
                {
                    for (int j = 0; j < 4; j++)//cloumns
                    {
                        double sum = 0;
                        for (int k = 0; k < 20; k++)//samples
                        {
                            sum += ((Data[k, i] - means[l][i]) * (Data[k, j] - means[l][j]));
                        }
                        covariances[l][i, j] = sum / 20;
                    }
                }
            }
        }

        Matrix<double> absMatrix(Matrix<double>M)
        {
            for(int i=0;i<M.RowCount;i++)
            {
                for(int j=0;j<M.ColumnCount;j++)
                {
                    if(M[i,j]<0)
                        M[i,j]*=-1;
                }
            }
            return M;
        }

         Matrix<double> sqrtMatrix(Matrix<double>M)
        {
            for(int i=0;i<M.RowCount;i++)
            {
                for(int j=0;j<M.ColumnCount;j++)
                {
                    M[i,j]=Math.Pow(M[i,j],0.5);
                }
            }
            return M;
        }

        public void classifier()
        {
            //likelihood
            Matrix<double> X = Matrix<double>.Build.Dense(4, 1);
            Matrix<double> tmp_vector = Matrix.Build.Dense(4, 1);
           
          
            for (int i = 0; i < 3; i++)
            {
                switch (i)
                {
                    case 0:
                        
                            for (int j = 20,k=0; j < 50; j++,k++)
                            {
                                for (int l = 0; l < 3; l++)
                                {
                                    double d = Math.Pow((2 * Math.PI), 2) * Math.Sqrt(covariances[l].Determinant());
                                    d = 1 / d;
                                    Matrix<double> tmpCovariences = Matrix.Build.Dense(4, 4);
                                  //  tmpCovariences=covariances[l].Multiply(Math.Pow((2 * Math.PI), 2));
                                    //exponant part
                                    X=Data.Row(j).ToColumnMatrix();//one row of samples 
                                    tmp_vector=(X-means[l].ToColumnMatrix());//X-Mi
                                    tmp_vector=tmp_vector.Transpose();//(X-Mi)t
                                    tmp_vector*=covariances[l].Inverse();
                                    tmp_vector *= (X - means[l].ToColumnMatrix());//X-Mi;
                                    tmp_vector*=-0.5;
                                    double d3 =(tmp_vector[0,0]);
                                    double d2 = Math.Exp(d3);
                                    d *= d2;
                                    likelihood[i][k, l] = d;
                                    //likelihood[i][k,l]=
                                }
                            }
                        
                        break;

                    case 1:
                        {
                            for (int j = 70,k=0; j < 100; j++,k++)
                            {
                                for (int l = 0; l < 3; l++)
                                {
                                    double d = Math.Pow((2 * Math.PI), 2) * Math.Sqrt(covariances[l].Determinant());
                                    d = 1 / d;
                                    Matrix<double> tmpCovariences = Matrix.Build.Dense(4, 4);
                                    //  tmpCovariences=covariances[l].Multiply(Math.Pow((2 * Math.PI), 2));
                                    //exponant part
                                    X = Data.Row(j).ToColumnMatrix();//one row of samples 
                                    tmp_vector = (X - means[l].ToColumnMatrix());//X-Mi
                                    tmp_vector = tmp_vector.Transpose();//(X-Mi)t
                                    tmp_vector *= covariances[l].Inverse();
                                    tmp_vector *= (X - means[l].ToColumnMatrix());//X-Mi;
                                    tmp_vector *= -0.5;
                                    double d3 = (tmp_vector[0, 0]);
                                    double d2 = Math.Exp(d3);
                                    d *= d2;
                                    likelihood[i][k, l] = d;
                                   
                                }
                            }
                        }
                        break;

                    case 2:
                        {
                            for (int j = 120,k=0; j < 150; j++,k++)
                            {
                                for (int l = 0; l < 3; l++)
                                {
                                    double d = Math.Pow((2 * Math.PI), 2) * Math.Sqrt(covariances[l].Determinant());
                                    d = 1 / d;
                                    Matrix<double> tmpCovariences = Matrix.Build.Dense(4, 4);
                                    //  tmpCovariences=covariances[l].Multiply(Math.Pow((2 * Math.PI), 2));
                                    //exponant part
                                    X = Data.Row(j).ToColumnMatrix();//one row of samples 
                                    tmp_vector = (X - means[l].ToColumnMatrix());//X-Mi
                                    tmp_vector = tmp_vector.Transpose();//(X-Mi)t
                                    tmp_vector *= covariances[l].Inverse();
                                    tmp_vector *= (X - means[l].ToColumnMatrix());//X-Mi;
                                    tmp_vector *= -0.5;
                                    double d3 = (tmp_vector[0, 0]);
                                    double d2 = Math.Exp(d3);
                                    d *= d2;
                                    likelihood[i][k, l] = d;
                                   
                                }
                            }
                        }
                        break;
                }

            }//end for (likelihood)

            //Evidence
            //for(int i=0;i<3;i++)//classes
            //{
            //    for(int j=0;j<30;j++)//samples
            //    {
            //        for(int l=0;l<3;l++)//features
            //        {
            //            evidence += likelihood[i][j, l] * (0.3);
            //        }
            //    }
            //}

            for (int i = 0; i < 3;i++ )//classes
            {
                for(int j=0;j<30;j++)
                {
                    double sum=0.0;
                    for(int k=0;k<3;k++)
                    {
                        sum += likelihood[i][j, k] * 0.3;
                    }
                    evidence[i][j] = sum;
                }
            }

                //posterior
                for (int i = 0; i < 3; i++)
                {
                    for (int j = 0; j < 30; j++)//rows
                    {
                        for (int l = 0; l < 3; l++)//columns
                        {
                            Posterior[i][j, l] = (((likelihood[i][j, l] * (0.3))) / evidence[i][j]);
                        }
                    }
                }
          
            //confusion matrix
            for(int i=0;i<3;i++)
            {
                for(int j=0;j<30;j++)
                {
                   
                    int tmp_index = Posterior[i].Row(j).MaximumIndex();
                    Confusion[i, tmp_index]++;
                }
            }
            percentage = ((Confusion.Diagonal().Sum())/90.0f)*100 ;

        }
    }
}
