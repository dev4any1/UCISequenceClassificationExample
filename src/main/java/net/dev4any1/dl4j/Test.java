package net.dev4any1.dl4j;

import java.util.Arrays;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.common.primitives.Pair;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import net.dev4any1.dl4j.LSTMSequenceModel.LstmConfig;

public class Test {

	/**
	 * Test assumed doing forecasts of the chart trend state. 
	 * The labels are implicitly specified in
	 * UCI data-set. Presuming each 100 rows are grouped by certain type.
	 * Types are {@link UCITimeSeriesCollectionParser.LSTMLabels}
	 */

	public static void main(String[] args) {
		int numLabels = UCITimeSeriesCollectionParser.LSTMLabels.length;
		int epoh = 28;
		int batch = 20;
		int hiddenSize = 100;
		test(batch, numLabels, epoh, hiddenSize);
	}

	private static int maxMatchCount = 0;
	private static String maxMatchResult = "";

	public static void test(int batch, int numLabels, int epoh, int hiddenSize) {
		LstmConfig config = new LstmConfig(batch, numLabels, epoh, hiddenSize);
		DatavecCollectionParser parser = new UCITimeSeriesCollectionParser(config);
		LSTMSequenceModel model = new LSTMSequenceModel(config);
		parser.parse(LSTMSequenceModel.class.getClassLoader().getResource("synthetic_control.data"));
		MultiLayerNetwork network = model.getNetworkTrained(parser.getTrainIterator(), parser.getTestIterator());
		parser.getTestIterator().reset();
		Evaluation eval = model.evaluate(parser.getTestIterator(), network);
		System.out.println(eval);
		int matchCount = 0;
		int[] sumsTotal = new int[numLabels];
		for (Pair<String, Integer> featureLabelPair : parser.getFeatureLabelPairs()) {
			String featureToForecast = featureLabelPair.getKey();
			int labelOfFeature = featureLabelPair.getValue();
			INDArray forecast = model.getForecast(network, featureToForecast.replace("\\s+", " "));
			INDArray flat = forecast.reshape(config.getNumLabelClasses(), 60);
			INDArray sums = flat.sum(1);
			for (int i = 0; i < sums.size(0); i++) {
				sumsTotal[i] += sums.getDouble(i);
			}
			System.out.println("l: " + labelOfFeature + " sums: " + sums.toString(numLabels, false, 4));
			int modelSwears = Nd4j.argMax(sums, 0).getInt(0);
			if (labelOfFeature == modelSwears) {
				matchCount++;
			}
		}

		if (matchCount > maxMatchCount) {
			maxMatchCount = matchCount;
			maxMatchResult = " Curent max match: " + matchCount + " e:" + epoh + " b:" + batch + " hs:" + hiddenSize;
		}
		System.out.println("[e: " + epoh + " b: " + batch + " hs: " + hiddenSize + "] matches: "
				+ matchCount + " Sums total " + Arrays.toString(sumsTotal) + maxMatchResult);
	}
}
