# UCISequenceClassificationExample

Sequence Classification Example Using a LSTM Recurrent Neural Network (deeplearning4j) to train/test/forecast the network model on subject: if the sequence of double is matching certain criteria.

Namely learns how to classify univariate time series as belonging to one of six categories. 
Categories are: Normal, Cyclic, Increasing trend, Decreasing trend, Upward shift, Downward shift. The <a href="https://github.com/deeplearning4j/deeplearning4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/modeling/recurrent/UCISequenceClassification.java">original</a> code was reworked to not use the file-system
		
#The Data is the UCI Synthetic Control Chart Time Series Data Set

<img src="https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/data.jpeg">

The
<a href="https://archive.ics.uci.edu/ml/datasets/Synthetic+Control+Chart+Time+Series">deatils</a>
are avaliable as well as the
<a href="https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/synthetic_control.data">raw raw dataset</a>


#The Network configuration: 

1 x input; init of weights - XAVIER; gradient update - NADAM; 

Layer1: LSTM activation - TANH; 

Layer2: RNN activation - SOFTMAX; loss function - MCXENT


