#include "getL.h"

double getLambda(std::vector<std::vector<double>> &a){
    int N = a.size();
    int k = 0;
    std::vector<double> X(N,0),Y(N,1.0);
    double max1 = 0.0;
    double max2 = 1.0*N;
    double m1 = 0.0; 
    double m2 = 0.0;
    double tmpfloat = 0.0;
    while ((fabs(max1 - max2) > 0.0001 ) || ( k < 3)){
        max2 = max1;
        m1 = m2;
        for(int i=0;i<N;++i){
            X[i] = 0.0;
            for(int j=0;j<N;++j)
                X[i] += a[i][j]*Y[j];            
        }
        tmpfloat = 0.0;
        for(int i=0;i<N;++i)
            if (X[i]> tmpfloat)
                tmpfloat = X[i];
        if (tmpfloat == 0.0)
            return  0.0;
        else{
            for(int i=0;i<N;++i)
                Y[i] = X[i] / tmpfloat;
            m2 = tmpfloat;
            max1 = sqrt(m1*m2);
            if (k > 5000) break;
            else k++;            
        }
    }
    return max1;    
}
