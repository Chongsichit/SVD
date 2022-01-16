import Jama.Matrix;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.*;

public class SVDandPCA {

    /*
    Primary Matrix: X, EigenValue Matrix: E, Right EigenVector Matrix: V, Left EigenVector Matrix:U
    SVD: X=UEV^T
    PCA: Y=XP --> Y=UEV^T P (V是XXT的沒有降維的EigenVector Matrix, P是已降維的EigenVector Matrix，所以降維後，P是等於V) --> Y=UE
    XP = UE
    1.去中心化
    2.求原矩陣的XXT(左)的E,U
    3.按XXT的大小排序U
    4.Y=UE
     */

    private static double[][] centralization (double[][] matrix){
        int n = matrix.length;
        int m = matrix[0].length;
        double [][] average0matrix = new double[n][m];
        double [] average = new double[m];
        double[] sum = new double[m];
        for ( int j = 0 ; j < m ; j++){  // j = columns
            for ( int i = 0 ; i < n ; i++){  // i = rows
                sum[j] += matrix[i][j];
            }
            average[j] = sum[j]/n;}
        for(int j = 0; j<m; j++){
            for(int i = 0 ; i < n ; i++){
                average0matrix[i][j]=matrix[i][j]-average[j];
            }
        }

        return average0matrix;
    } //使樣本均值為零，方便計算

    private static Matrix covarianceMatrix(double[][] matrix){
        int m = matrix[0].length; //m為總變量數 n*m matrix
        int n = matrix.length; //n為總行數
        double [][] result = new double[n][n];
        for ( int i = 0; i < n; i++){
            for ( int j = 0; j < n; j++){
                double temp = 0;
                for ( int k = 0; k < m; k++){
                    temp += matrix[i][k] * matrix[j][k];
                }
                result[i][j] = temp / (n-1) ;
            }
        }

        return new Matrix(result);
    }//求出協方差矩陣

    private static Matrix Esqrt (Matrix matrix){
        Matrix matrix1 = matrix.eig().getD();
        double[][] result = matrix1.getArray();
        for (int i =0; i < result.length ; i++){
            for (int j =0; j< result[i].length; j++){
                if ( i == j){
                    result[i][j]= Math.sqrt(result[i][j]);
                }
            }
        }

        return new Matrix(result);
    }

    private static Matrix UReduction(Matrix eigenValueM, Matrix eigenVectorM){
        double[][] eigenValue = eigenValueM.getArray();
        double[][] eigenVectorTrans = eigenVectorM.transpose().getArray();
        double[][] result = new double[2][8];
        List<Double> eigenValueList = new ArrayList<>();
        HashMap<Double,double[]> hashMap = new HashMap<>();

        for(int i = 0; i < eigenValue.length ; i++){
            for( int j=0; j< eigenValue[i].length; j++){
                if(i==j){
                    eigenValueList.add(eigenValue[i][j]);
                    double[] eigenVector = eigenVectorTrans[i];
                    hashMap.put(eigenValue[i][j],eigenVector);
                }
            }
        }

        Collections.reverse(eigenValueList);

        for(int i = 0; i < result.length; i++){
            result[i] = hashMap.get(eigenValueList.get(i));
        }

        return new Matrix(result).transpose();
    }

    private static Matrix EReduction(Matrix eigenValueM){
        int limit = 2;
        double[][] eigenValue = eigenValueM.getArray();
        List<Double> temp = new ArrayList<>();
        double[][] result = new double[limit][limit];
        for(int i = 0; i < eigenValue.length; i++){
            temp.add(eigenValue[i][i]);
        }
        Collections.reverse(temp);

        for(int i =0;i<result.length;i++){
            for (int j=0;j<result[i].length;j++){
                if(i==j) result[i][j]=temp.get(i);
                        else result[i][j]=0;
            }
        }


        return new Matrix(result);
    }

    private static void outputData(String path, double[][] matrix) throws IOException {
        FileOutputStream fileOutputStream = new FileOutputStream(path);
        OutputStreamWriter outputStreamWriter = new OutputStreamWriter(fileOutputStream);
        BufferedWriter bufferedWriter = new BufferedWriter(outputStreamWriter);
        for(int i = 0; i < matrix.length; i++){
//            bufferedWriter.write(i+":");
            for (int j =0;  j<matrix[i].length; j++){
                bufferedWriter.write(matrix[i][j] +",");
            }
            bufferedWriter.newLine();
            bufferedWriter.flush();
        }
    }

    public static void main(String[] args) throws IOException {

        String path ="C:\\Users\\ASUS\\Desktop\\mnist\\SVDandPCA";

        double[][] data = {{7,4,3,4},{4,1,8,3},{6,3,5,2},{8,3,2,10},{4,5,0,9},{1,3,2,5},{6,6,3,2},{8,3,3,6}};
        double[][] dataMatrix = centralization(data);
        Matrix primaryXtransposed = covarianceMatrix(dataMatrix);
        Matrix E = primaryXtransposed.eig().getD();
        Matrix U = primaryXtransposed.eig().getV();


        Matrix EReduct = EReduction(E);
        Matrix UReduct = UReduction(E,U);
        Matrix Esqrt = Esqrt(EReduct);

        double [][] outputData = UReduct.times(Esqrt).getArray();
        for(int i=0;i<outputData.length;i++){
            System.out.println(Arrays.toString(outputData[i]));
        }
        outputData(path,outputData);
    }
}
