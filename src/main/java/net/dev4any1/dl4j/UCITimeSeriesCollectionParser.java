package net.dev4any1.dl4j;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionSequenceRecordReader;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import net.dev4any1.dl4j.LSTMSequenceModel.LstmConfig;

public class UCITimeSeriesCollectionParser implements DatavecCollectionParser {

	public static String[] LSTMLabels = { "Normal", "Cyclic", "Increasing Trend", "Decreasing Trend", "UpwardShift",
			"DownwardShift" };

	private static final Logger log = Logger.getLogger(UCITimeSeriesCollectionParser.class.getName());

	private List<Pair<String, Integer>> featureLabelPairs = new ArrayList<Pair<String, Integer>>();

	private DataSetIterator trainDsIterator;

	private DataSetIterator testDsIterator;

	private LstmConfig config;

	public UCITimeSeriesCollectionParser(LstmConfig config) {
		this.config = config;
	}

	public DataSetIterator getTrainDsIterator() {
		return trainDsIterator;
	}

	public DataSetIterator getTestDsIterator() {
		return testDsIterator;
	}

	@Override
	public List<Pair<String, Integer>> getFeatureLabelPairs() {
		return featureLabelPairs;
	}

	/**
	 * This dataset contains 600 examples of control charts generated There are six
	 * different classes (features) of control charts: Normal, Cyclic, Increasing
	 * trend, Decreasing trend, Upward shift, Downward shift The data is stored in
	 * an ASCII file, 600 rows, 60 columns, with a single chart per line. The
	 * classes are organized as follows: 1-100 Normal 101-200 Cyclic 201-300
	 * Increasing trend 301-400 Decreasing trend 401-500 Upward shift 501-600
	 * Downward shift
	 */

	@Override
	public void parse(URL source) {
		final int featureSize = 100;
		List<String> trainFeaturesList = new ArrayList<String>();
		List<String> trainLabelsList = new ArrayList<String>();
		List<String> testFeatureList = new ArrayList<String>();
		List<String> testLabelsList = new ArrayList<String>();
		if (source == null) {
			try {
				source = new URL(
						"https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/synthetic_control.data");
			} catch (MalformedURLException e) {
				log.log(Level.SEVERE, "unable to initializeLstmSmDataData due to MalformedURLException", e);
			}
		}
		try (BufferedReader reader = new BufferedReader(
				new InputStreamReader(source.openStream(), StandardCharsets.UTF_8))) {
			List<String> data = reader.lines().collect(Collectors.toList());
			String[] lines = data.toArray(new String[data.size()]);

			int lineCount = 0;
			new ArrayList<>(lines.length);
			// Labels: first 100 lines are label 0, second 100 are label 1, and so on
			for (String line : lines) {
				featureLabelPairs.add(new Pair<>(line, lineCount++ / featureSize));
			}
			// Randomize and do a train/test split:
			Collections.shuffle(featureLabelPairs, new Random(12345));
			for (int i = 0; i < featureLabelPairs.size(); i++) {
				trainFeaturesList.add(featureLabelPairs.get(i).getKey());
				trainLabelsList.add(String.valueOf(featureLabelPairs.get(i).getValue()));
				testFeatureList.add(featureLabelPairs.get(i).getKey());
				testLabelsList.add(String.valueOf(featureLabelPairs.get(i).getValue()));
			}
		} catch (IOException e) {
			log.log(Level.SEVERE, "unable to initializeLstmSmDataData due to IOException", e);
		}
		this.trainDsIterator = new SequenceRecordReaderDataSetIterator(parse(trainFeaturesList), parse(trainLabelsList),
				config.getMiniBatchSize(), config.getNumLabelClasses(), false,
				SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);
		this.testDsIterator = new SequenceRecordReaderDataSetIterator(parse(testFeatureList), parse(testLabelsList),
				config.getMiniBatchSize(), config.getNumLabelClasses(), false,
				SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);
	}

	/*
	 * This method transforms input array of rowStrings to the 3d List
	 * with @Writable for @CollectionSequenceRecordReader as required by
	 * org.deeplearning4j.datasets.datavec rawStrings might contain a list of
	 * timeSeries (features) as well as their labels in case of features we will
	 * have a list of strings with space-separated doubles; in case of labels we
	 * will have a string with a single int (double) label mark
	 */
	private SequenceRecordReader parse(List<String> rawStrings) {
		double max = Double.MIN_VALUE;
		double min = Double.MAX_VALUE;
		List<List<List<Writable>>> top = new ArrayList<>();
		try {
			for (int i = 0; i < rawStrings.size() - 1; i++) {
				List<List<Writable>> mid = new ArrayList<>();
				// if we have a feature rawString
				if (rawStrings.get(i).contains(" ")) {
					String[] values = rawStrings.get(i).split("\\s+");
					for (String v : values) {
						List<Writable> bottom = new ArrayList<>();

						double d = Double.parseDouble(v);
						max = d > max ? d : max;
						min = d < min ? d : min;
						bottom.add(new DoubleWritable(d));
						mid.add(bottom);
					}
				} else {
					// otherwise it is a label value
					List<Writable> bottom = new ArrayList<>();
					bottom.add(new DoubleWritable(Double.parseDouble(rawStrings.get(i))));
					mid.add(bottom);
				}
				top.add(mid);
			}
			if (max != Double.MIN_VALUE)
				log.info("feature range in dataset [" + min + " .. " + max + "]");
			log.info("top: [" + top.size() + "][" + top.get(0).size() + "][" + top.get(0).get(0).size() + "]");
			return new CollectionSequenceRecordReader(top);
		} catch (NumberFormatException nfe) {
			log.log(Level.SEVERE,
					"Unable to parse dataset due to NumberFormatException, expecting CSV inout of doubles with space delimeter ",
					nfe);
			return null;
		}
	}

	@Override
	public DataSetIterator getTrainIterator() {
		return getTrainDsIterator();
	}

	@Override
	public DataSetIterator getTestIterator() {
		return getTestDsIterator();
	}
}
