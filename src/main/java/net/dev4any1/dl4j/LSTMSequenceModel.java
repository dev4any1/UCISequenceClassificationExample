package net.dev4any1.dl4j;

import java.util.Arrays;
import java.util.logging.Logger;

import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Sequence Classification Example Using a LSTM Recurrent Neural Network
 *
 * This learns how to classify univariate time series as belonging to one of six
 * categories. Categories are: Normal, Cyclic, Increasing trend, Decreasing
 * trend, Upward shift, Downward shift
 *
 * Data is the UCI Synthetic Control Chart Time Series Data Set
 * 
 * @see <a href=
 *      "https://archive.ics.uci.edu/ml/datasets/Synthetic+Control+Chart+Time+Series">details</a>
 * @see <a href=
 *      "https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/synthetic_control.data">data</a>
 * @see <a href=
 *      "https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/data.jpeg">visualisation</a>
 */

public class LSTMSequenceModel {

	private static final Logger log = Logger.getLogger(LSTMSequenceModel.class.getName());
	private LstmConfig config;
	private DataNormalization normalizer = new NormalizerStandardize();
	private MultiLayerNetwork net;

	public LSTMSequenceModel(LstmConfig config) {
		this.config = config;
		net = configAndInitNetwork();
	}

	private MultiLayerNetwork configAndInitNetwork() {
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(123).weightInit(WeightInit.XAVIER)
				.updater(new Nadam())
				// Not always required, but helps with this data set
				.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
				.gradientNormalizationThreshold(0.5).list()
				.layer(new LSTM.Builder().activation(Activation.TANH).nIn(1).nOut(config.getHiddenStatesSize()).build())
				.layer(new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)
						.nIn(config.getHiddenStatesSize()).nOut(config.getNumLabelClasses()).build())
				.build();
		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		return net;
	}

	public MultiLayerNetwork getNetworkTrained(DataSetIterator trainData, DataSetIterator testData) {
		if (trainData == null)
			throw new IllegalArgumentException("Invalid input: test trainData is null");
		if (testData == null)
			throw new IllegalArgumentException("Invalid input: test datasetIterator is null");
		log.info(Nd4j.getExecutioner().getEnvironmentInformation().toString());
		// Collect training data statistics
		normalizer.fit(trainData);
		trainData.reset();
		// Use previously collected statistics to normalize on-the-fly. Each DataSet
		// returned by 'trainData' iterator will be normalized
		trainData.setPreProcessor(normalizer);
		// NB we are using the exact same normalization process as the training data
		testData.setPreProcessor(normalizer);
		final long now = System.currentTimeMillis();
		// log.info("Start training, epochs: " + config.getNumEpochs() + ", minibatch: "
		// + config.getMiniBatchSize());
		net.setListeners(new ScoreIterationListener(20), new EvaluativeListener(testData, 1, InvocationType.EPOCH_END));
		net.fit(trainData, config.getNumEpochs());
		log.info("Fit took: " + (System.currentTimeMillis() - now) / 1000 + " sec.");
		return net;
	}

	/**
	 * predicting class index with the highest probability
	 * 
	 * @param network
	 * @param testFeature
	 * @return API @Forecast
	 */
	public INDArray getForecast(MultiLayerNetwork net, String testFeature) {
		// Now predict using the provided testFeature
		if (net == null)
			throw new IllegalArgumentException("Invalid input: network is null");
		if (testFeature == null)
			throw new IllegalArgumentException("Invalid input: testFeature is null");
		final double[] doubles = Arrays.stream(testFeature.split("\\s+")).mapToDouble(v -> Double.valueOf(v)).toArray();
		final long[] shape = { 1, 1, doubles.length };
		INDArray input = Nd4j.create(doubles, shape, 'c');
		normalizer.transform(input);
		return net.output(input, false);
	}

	public Evaluation evaluate(DataSetIterator testData, MultiLayerNetwork net) {
		// Now predict using the provided testFeature
		if (net == null)
			throw new IllegalArgumentException("Invalid input: network is null");
		if (testData == null)
			throw new IllegalArgumentException("Invalid input: test datasetIterator is null");
		log.info("Evaluating...");
		return net.evaluate(testData);
	}

	public static class LstmConfig {
		private int miniBatchSize;
		private int hiddenStatesSize;
		private int numLabelClasses;
		private int numEpochs;

		public LstmConfig(int miniBatchSize, int numLabelClasses, int numEpochs, int hiddenStatesSize) {
			this.miniBatchSize = miniBatchSize;
			this.numEpochs = numEpochs;
			this.numLabelClasses = numLabelClasses;
			this.hiddenStatesSize = hiddenStatesSize;
		}

		public int getMiniBatchSize() {
			return miniBatchSize;
		}

		public int getNumLabelClasses() {
			return numLabelClasses;
		}

		public int getNumEpochs() {
			return numEpochs;
		}

		public int getHiddenStatesSize() {
			return hiddenStatesSize;
		}
	}
}
