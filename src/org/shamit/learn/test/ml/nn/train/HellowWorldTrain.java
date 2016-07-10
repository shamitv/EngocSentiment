package org.shamit.learn.test.ml.nn.train;

import java.util.List;

import org.shamit.learn.test.ml.nn.data.DataLoader;
import org.shamit.learn.test.ml.nn.data.Sentence;

public class HellowWorldTrain {

	public static void main(String[] args) throws Exception{
		int maxLen=200;
		DataLoader dl = new DataLoader();
		List<Sentence> sents = dl.getSentences();
		System.out.println(sents.size() + " sentences loaded.");
		double data[][]=new double[sents.size()][];
		double labels[][]=new double[sents.size()][];
		//
		// Initialize training data
		//
		int i=0;
		for(Sentence s:sents){
			data[i]=s.getEncodedText(maxLen);
			labels[i]=new double[1];
			double score=0;
			if(s.isSentimentPositive()){
				score=1;
			}
			labels[i][0]=score;
			i++;
		}
		System.out.println("Training data prepared.");
		
	}

}
