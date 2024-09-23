package net.dev4any1.dl4j;

import java.net.URL;
import java.util.List;

import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public interface DatavecCollectionParser {

	/**
	 * The method of fetching the data from the given {@link java.net.URL}
	 */
	public void parse(URL source);

	/**
	 * The method of getting the raw time-series data (CSV String) grouped by the Label value (Integer) 
	 * @return
	 */

	public List<Pair<String, Integer>> getFeatureLabelPairs();
	/**
	 * Get training data prepared for the model
	 * @return {@link org.nd4j.linalg.dataset.api.iterator.DataSetIterator}
	 */
	public DataSetIterator getTrainIterator();

	/**
	 * Get verification data prepared for the model
	 * @return {@link org.nd4j.linalg.dataset.api.iterator.DataSetIterator}
	 */
	public DataSetIterator getTestIterator();
}
