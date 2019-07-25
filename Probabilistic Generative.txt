#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <math.h>

#define PI 3.141592

using namespace std;
using namespace Eigen;

typedef vector<float> vf;
typedef vector<vector<float> > vvf;

int main()
{
    ifstream inputFile;
    string singleLine;
    float attr[4];
    bool check;
    char commas[4];

    Matrix4f S1, S2;
    vvf Zeros, Ones;
    Vector4f u1;
    Vector4f u2;

    for (int i=0;i<4;i++)
    {
        u1(i,0)=0;
        u2(i,0)=0;
    }

    inputFile.open("train.txt");
    // training the data
    while(inputFile.good()){

        getline(inputFile,singleLine);
        stringstream s(singleLine);
        if(singleLine.length() < 3){break;}
        s  >> attr[0] >> commas[0] >> attr[1] >> commas[1]  >> attr[2] >> commas[2]  >> attr[3] >> commas[3] >> check;

        vf temp;
        for (int i=0;i<4;i++)
        {
            temp.push_back(attr[i]);
        }

        if(check == 0){
            Zeros.push_back(temp);
            for(int i = 0; i<4; ++i)
            {
                u1(i,0) += attr[i];
            }
        }
        else{
            Ones.push_back(temp);
            for(int i = 0; i<4; ++i)
            {
                u2(i,0) += attr[i];
            }
        }
    }
    inputFile.close();
    int len_0 = Zeros.size();
    int len_1 = Ones.size();

    // dividing by no. of objects of each class to get the average
    for(int i = 0; i<4; ++i)
    {
        u1(i,0) /= len_0;
        u2(i,0) /= len_1;
    }
    // initialization of S1 and S2 with zeros

    for (int i=0;i<4;i++)
    {
        for (int j=0;j<4;j++)
        {
            S1(i,j) = 0;
            S2(i,j) = 0;
        }
    }

    // finding S1 and S2
    // for zeros
    for(int i = 0; i<len_0; ++i)
    {
        Vector4f single;
        for (int j=0;j<4;j++)
        {
            single(j,0) = Zeros[i][j];
        }
        for(int j = 0; j<4; ++j)
        {
            single(j,0) -= u1(j,0);
        }

        // single * single(transpose)
        for (int j=0;j<4;j++)
        {
            for (int k=0;k<4;k++)
            {
                S1 += single*single.transpose();
            }
        }
    }

    // for ones
    for(int i = 0; i<len_1; ++i)
    {
        Vector4f single;
        for (int j=0;j<4;j++)
        {
            single(j,0) = Ones[i][j];
        }
        for(int j = 0; j<4; ++j)
        {
            single(j,0) -= u2(j,0);
        }

        // single * single(transpose)
        for (int j=0;j<4;j++)
        {
            for (int k=0;k<4;k++)
            {
                S2 += single*single.transpose();
            }
        }
    }
    Matrix4f S;
    S = S1 + S2;
    S /= len_0+len_1;

    Matrix4f s_inv = S.inverse();

    // testing the data
    inputFile.open("test.txt");
    int p_p=0,p_n=0,n_p=0,n_n=0;
    while(inputFile.good()){
        getline(inputFile,singleLine);
        stringstream s(singleLine);
        if(singleLine.length() < 3){break;}             // condition for termination
        s  >> attr[0] >> commas[0] >> attr[1] >> commas[1]  >> attr[2] >> commas[2]  >> attr[3] >> commas[3] >> check;

        Vector4f x;
        for (int i=0;i<4;i++)
        {
            x(i,0) = attr[i];
        }
        double p1,p2;
        int d = 4;
        double sqrtdet = sqrt(S.determinant());
        double expo = exp((-0.5)*((x-u1).transpose())*(s_inv)*(x-u1));
        p1 = (1/pow(2*PI,d/2))*(1/sqrtdet)*expo;                            // prob of zero
        expo = exp((-0.5)*((x-u2).transpose())*(s_inv)*(x-u2));
        p2 = (1/pow(2*PI,d/2))*(1/sqrtdet)*expo;                            // prob of one
        if (p1>p2)
        {
            if (check==0)
            {
                n_n++;
            }
            else
            {
                n_p++;
            }
        }
        else
        {
            if (check==1)
            {
                p_p++;
            }
            else
            {
                p_n++;
            }
        }
    }
    // final output
    double precision = ((double)p_p/(p_p+p_n))*100;
    double recall = ((double)p_p/(p_p+n_p))*100;
    cout << '\t' << '\t' << "predicted" << endl;
    cout <<  '\t' << '\t'<< "n"<< '\t'<<"p" << endl;
    cout << "actual" << '\t'<<  "n" << '\t' << n_n << '\t' << p_n << endl;
    cout << '\t' <<  "p" << '\t' << n_p << '\t' << p_p << endl;
    cout << "Precision = " << precision << "%" << endl;
    cout << "Recall = " << recall<< "%" << endl;
    inputFile.close();
    return 0;
}
