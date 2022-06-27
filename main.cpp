#include <iostream>
#include <cmath>
#include <vector>
#include <omp.h>
#include <fstream>
#include <bits/stdc++.h>
#include <time.h>


using namespace std;


#define M  10000


//--------------- define find min reduction -------------
typedef std::pair<unsigned int, float> IndexValuePair;

IndexValuePair myMin(IndexValuePair a, IndexValuePair b){
    return a.second < b.second ? a : b;
}

#pragma omp declare reduction \
        (minPair:IndexValuePair:omp_out=myMin(omp_out, omp_in)) \
        initializer(omp_priv = IndexValuePair(0, 999999))
//-------------------------------------------------------


// Class simplex contains multiple parallel function 
class Simplex{

    private:
        int rows, cols;
        //stores coefficients of all the variables
        std::vector <std::vector<float> > A;
        //stores constants of constraints
        std::vector<float> B;
        //stores the coefficients of the objective function
        std::vector<float> C;
        std::vector<float> Cp;
        //stores the index of variables
        std::vector<float> Bvar;
        int vars ;
        float maximum;

        bool isUnbounded;

    public:
        Simplex(std::vector <std::vector<float> > matrix,std::vector<float> b ,std::vector<float> c,
                            std::vector<float> bvar, int invars){
            maximum = 0;
            isUnbounded = false;
            rows = matrix.size();
            cols = matrix[0].size();
            A.resize( rows , vector<float>( cols , 0 ) );
            B.resize(b.size());
            Bvar.resize(b.size());
            C.resize(c.size());
            Cp.resize(c.size());
            vars = invars;
            for(int i= 0;i<rows;i++){             //pass A[][] values to the metrix
                for(int j= 0; j< cols;j++ ){
                    A[i][j] = matrix[i][j];

                }
            }

            for(int i=0; i< c.size() ;i++ ){      //pass c[] values to the B vector
                C[i] = c[i] ;
                Cp[i] = c[i] ;
            }
            for(int i=0; i< b.size();i++ ){      //pass b[] values to the B vector
                B[i] = b[i];
                Bvar[i] = bvar[i] ;
            }
        }

        bool simplexAlgorithmCalculataion(){
            //check whether the table is optimal,if optimal no need to process further
            if(checkOptimality()==true){
			    return true;
            }

            //find the column which has the pivot.The least coefficient of the objective function(C array).
            int pivotColumn = findPivotColumn();


            if(isUnbounded == true){
                cout<<"Error unbounded"<<endl;
			    return true;
            }

            //find the row with the pivot value.The least value item's row in the B array
            int pivotRow = findPivotRow(pivotColumn);

            cout<<"-----------------"<<endl;
            cout<<pivotRow<<"---" << pivotColumn <<endl;
            
            Bvar[pivotRow] = pivotColumn;

            //form the next table according to the pivot value
            doPivotting(pivotRow,pivotColumn);

            return false;
        }

        //When the answare in optimal , there is no better answer 
        //so we have to stop iterating on the tabluea
        bool checkOptimality(){
             //if the table has further negative constraints,then it is not optimal
            bool isOptimal = false;
            int positveValueCount = 0;

            //check if the coefficients of the objective function are negative
            for(int i=0; i<C.size();i++){
                float value = C[i];
                if(value >= 0){
                    positveValueCount++;
                }
            }
            //if all the constraints are positive now,the table is optimal
            if(positveValueCount == C.size()){
                isOptimal = true;
                print();
            }
            return isOptimal;
        }

        void doPivotting(int pivotRow, int pivotColumn){

            float pivetValue = A[pivotRow][pivotColumn];//gets the pivot value

            float pivotRowVals[cols];//the column with the pivot

            float pivotColVals[rows];//the row with the pivot

            float rowNew[cols];//the row after processing the pivot value

            maximum = maximum - (C[pivotColumn]*(B[pivotRow]/pivetValue));  //set the maximum step by step

            #pragma omp parallel
            {
            //get the row that has the pivot value
             #pragma omp for
             for(int i=0;i<cols;i++){
                pivotRowVals[i] = A[pivotRow][i];
             }
             //get the column that has the pivot value
             #pragma omp for
             for(int j=0;j<rows;j++){
                pivotColVals[j] = A[j][pivotColumn];
            }

            //set the row values that has the pivot value divided by the pivot value and put into new row
             #pragma omp parallel
             for(int k=0;k<cols;k++){
                rowNew[k] = pivotRowVals[k]/pivetValue;
             }

            }

            B[pivotRow] = B[pivotRow]/pivetValue;


             //process the other coefficients in the A array by subtracting
             for(int m=0;m<rows;m++){
                //ignore the pivot row as we already calculated that
                if(m !=pivotRow){
                    for(int p=0;p<cols;p++){
                        float multiplyValue = pivotColVals[m];
                        A[m][p] = A[m][p] - (multiplyValue*rowNew[p]);
                    }

                }
             }

            //process the values of the B array
            #pragma omp parallel for shared(pivotRow, B) 
            for(int i=0;i<B.size();i++){
                if(i != pivotRow){
                    float multiplyValue = pivotColVals[i];
                    B[i] = B[i] - (multiplyValue*B[pivotRow]);

                }
            }
                //the least coefficient of the constraints of the objective function
                float multiplyValue = C[pivotColumn];
                //process the C array
                 for(int i=0;i<C.size();i++){
                    C[i] = C[i] - (multiplyValue*rowNew[i]);

                }


             //replacing the pivot row in the new calculated A array
             #pragma omp parallel for shared(A, rowNew)
             for(int i=0;i<cols;i++){
                A[pivotRow][i] = rowNew[i];
             }


        }

        //print the current A array
        void print(){
            for(int i=0; i<rows;i++){
                for(int j=0;j<cols;j++){
                    cout<<A[i][j] <<" ";
                }
                cout<<""<<endl;
            }
            cout<<""<<endl;
        }

        //find the least coefficients of constraints in the objective function's position
        int findPivotColumn(){
            
            IndexValuePair minValueIndex(0, C[0]);

            #pragma omp parallel for reduction(minPair:minValueIndex)
            for(int i = 0; i < C.size(); i++){

                if(C[i] < minValueIndex.second){
                    minValueIndex.first = i;
                    minValueIndex.second = C[i];
                }
            }

            return minValueIndex.first;

        }

        //find the row with the pivot value.The least value item's row in the B array
        int findPivotRow(int pivotColumn){
            float positiveValues[rows];
            std::vector<float> result(rows,0);
            //float result[rows];
            int negativeValueCount = 0;

            #pragma omp parallel for \
                        shared(rows ,negativeValueCount, pivotColumn)
            for(int i=0;i<rows;i++){
                if(A[i][pivotColumn]>0){
                    positiveValues[i] = A[i][pivotColumn];
                }
                else{
                    positiveValues[i]=0;
                    #pragma omp critical
                    {
                        negativeValueCount+=1;
                    }
                }
            }
            //checking the unbound condition if all the values are negative ones
            if(negativeValueCount==rows){
                isUnbounded = true;
            }
            else{
                #pragma omp parallel for shared(result , positiveValues)
                for(int i=0;i<rows;i++){
                    float value = positiveValues[i];
                    if(value>0){
                        result[i] = B[i]/value;

                    }
                    else{
                        result[i] = 0;
                    }
                }
            }

           
            IndexValuePair minValueIndex(0, 9999999);

            #pragma omp parallel for reduction(minPair:minValueIndex)
            for(int i = 0; i < result.size(); i++){

                if(result[i] > 0 && result[i] < minValueIndex.second){
                    minValueIndex.first = i;
                    minValueIndex.second = result[i];
                }
            }

            return minValueIndex.first;
        }

        void CalculateSimplex(){
            bool end = false;

            cout<<"initial array(Not optimal)"<<endl;
            print();


            for (int i = 0 ; i<Bvar.size(); i++){
                int index = Bvar[i];
                if (index < C.size() && C[index] != 0){
                    float ratio = C[index] / A[i][index];
                    for(int j = 0 ; j < C.size() ; j++){
                        C[j] = C[j] - ratio * A[i][j];
                    }
                }
            }

            cout<<" "<<endl;
            cout<<"array after first iteration"<<endl;
            for(int i = 0 ; i < C.size() ; i++){
                cout<<C[i]<<" ";
            }
            cout<<endl;
            print();

            cout<<" "<<endl;
            cout<<"final array(Optimal solution)"<<endl;


            while(!end){

                bool result = simplexAlgorithmCalculataion();

                cout<<"----------"<<endl;
                if(result==true){

                    end = true;

                    }
            }
            cout<<"Answers for the Constraints of variables : "<<rows<<" : "<<endl;
            for(int i = 0 ; i<B.size() ; i++){
                cout<<"variable "<<Bvar[i]<<" : "<< B[i]<<endl;
            }

            // cout<<maximum<<endl;
            maximum = 0;
            for (int i = 0 ; i < B.size() ; i++){
               if (Bvar[i]<vars){
                    maximum += Cp[Bvar[i]]*B[i]* -1;
               }
           }

            
            cout<<maximum<<endl;
        }

};

std::vector<std::string> simple_tokenizer(string s)
{
    std::vector<std::string> result;
    stringstream ss(s);
    string word;
    while (ss >> word) {
        result.push_back(word);
    }
    return result;
}

int main()
{

    #ifdef _OPENMP
    cout<<"openmp exist"<<endl;
    #endif


    int vars = 8;
    int constarints = 8;
    //C array is the variables Coefficient in objective funtion
    float C[]= {-5,-10 , -8 , -2 , -3 , 5 , -9 , 19};
    //B array is the right side of the constraints
    float B[]={60,72,100 , 49 , 90 , 54 , 69, 43}; 
    //S array is the sign of the constraints
    // 1 means <=  and -1 means >= and 0 means = for each constrants
    float S[]={1 ,1 ,1 , -1 , 0 , -1 , 1 , 0}; 

    /*
    matrix a is the matrix of constrants
    note that we have to change this matrix to an tableau matrix since
    we may need to add some new columns
    */

    float a[constarints][vars] = {
                   { 3,  5, 2 , 4 ,3 ,2 ,5 ,8},
                   { 4,  4, 4 , 6 ,3 ,4, 2 ,5},
                   { 2,  4, 5 , 8, 2 ,6 ,8 ,2},
                   { 5,  2, 9 , 4, 4 , 7 ,3 ,9},
                   { 2,  8, 6 , 3, 6 , 4, 5, 7},
                   { 8,  3, 4 , 3, 8 , 4, 3, 5},
                   { 4,  3, 6 , 8, 6 , 3, 5, 2},
                   { 5,  2, 6 , 6, 4 , 3, 2, 5}
                };
   

    int colSizeA = sizeof(a[0])/sizeof(a[0][0]);
    int rowSizeA = sizeof(B)/sizeof(B[0]);
    
    int colSizeTablu = colSizeA + 2*rowSizeA; 
    int rowSizeTablu = rowSizeA; 

    //vec2D is the tableau
    std::vector <std::vector<float> > vec2D(rowSizeA, std::vector<float>(colSizeTablu, 0));

    //b is as same as B but we use it for the tableau
    std::vector<float> b(rowSizeTablu,0);
    std::vector<float> c(colSizeTablu,0);
    std::vector<float> bvar(rowSizeTablu,0);



    //filling the tableau
    for(int i=0;i<rowSizeA;i++){ 
        for(int j=0; j<colSizeTablu;j++){
            if(j<colSizeA){
                vec2D[i][j] = a[i][j];
            }
            else{
                if(S[i] == 1){
                    if(j-colSizeA == i){
                        vec2D[i][j] = 1;
                    }
                    else{
                        vec2D[i][j] = 0;
                    }
                
                }else if(S[i] == 0){
                    if(j- colSizeA - rowSizeA == i){
                        vec2D[i][j] = 1;
                    }
                    else{
                        vec2D[i][j] = 0;
                    }
                
                }else if(S[i]== -1){
                    if(j-colSizeA == i){
                        vec2D[i][j] = -1;
                    }
                    else if(j- colSizeA -rowSizeA == i){
                        vec2D[i][j] = 1;
                    }
                    else{
                        vec2D[i][j] = 0;
                    }
                }
            }
        }
    }

    for(int i=0;i<rowSizeTablu;i++){
        b[i] = B[i];
    }


    for(int j=0 ; j<sizeof(B)/sizeof(B[0]);j++){

        if (S[j] == 1){
            bvar[j] = colSizeA + j;
        }else if (S[j] == 0){
            bvar[j] = colSizeA + rowSizeA + j;
        }else{
            bvar[j] = colSizeA + rowSizeA + j;
        }
    }
 
    for(int i=0;i<colSizeTablu;i++){
        if(i < colSizeA){
            c[i] = C[i];
        }else if(i < colSizeA + rowSizeA){
            c[i] = 0;
        }else if(S[i-colSizeA-rowSizeA] <= 0){
            c[i] = M;
        }else{
            c[i] = 0;
        }
    }


    clock_t tStart = clock();

    // hear the make the class parameters with A[m][n] vector b[] vector and c[] vector
    Simplex simplex(vec2D,b,c,bvar , vars);
    simplex.CalculateSimplex();

    printf("Time taken: %.6fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);


    return 0;
}