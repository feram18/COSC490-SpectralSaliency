
Dataset  
  -
  Download the dataset from this link  
  http://opendata.deepsig.io/datasets/2016.10/RML2016.10a.tar.bz2  
  Then extract the .pkl file and put it with the same folder as .py files



Installation  
  -
Make sure to have:  
    - Python version 3.5–3.7  
    - pip 19.0 or later  

**_Installation To Run on CPU_**  
  -
open cmd then   
`pip install tensorflow==2.0.0`

 

**_Installation To Run on GPU_**  
  -
Use this link to follow the steps to  install the required softwares and to setup on windows
  - https://www.tensorflow.org/install/gpu    
Make sure when installing tensorflow using pip, use this command  
`pip install tensorflow-gpu`


---------------------------------------
---------------------------------

After finished installing,  
To run RNN model using GPU, follow this step:

  - uncomment CuDNNLSTM in line 8 at the end
  - comment out from line 49 - 62
  - uncommet from line 64 - 77
  
No need to do those if running normally with cpu


------
For RNN
  -
Change the number in the first parameter of "LSTM" (line 51, 54) and/or  the number in "Dropout" (line 52, 55, 58) to experiment and get different accuracy
  

