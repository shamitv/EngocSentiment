package org.shamit.learn.test.ml.nn.train;

import java.text.DecimalFormat;
import java.util.List;
import java.util.Random;

import org.encog.engine.network.activation.ActivationTANH;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.data.basic.BasicNeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
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
			labels[i]=s.getEncodedLabel();
			i++;
		}
		System.out.println("Training data prepared.");
		MLDataSet trainingSet = new BasicMLDataSet(data, labels);
		BasicNetwork network = new BasicNetwork();
		network.addLayer(new BasicLayer(null,true,maxLen));
		network.addLayer(new BasicLayer(new ActivationTANH(),true,maxLen));
		network.addLayer(new BasicLayer(new ActivationTANH(),true,maxLen));
		network.addLayer(new BasicLayer(new ActivationTANH(),true,maxLen/2));
		network.addLayer(new BasicLayer(new ActivationTANH(),true,maxLen/(2*2)));
		network.addLayer(new BasicLayer(new ActivationTANH(),true,maxLen/(2*2*2)));
		network.addLayer(new BasicLayer(new ActivationTANH(),false,1));
		network.getStructure().finalizeStructure();
		network.reset();
		
		System.out.println("Network created. Structure is :: \n\t"+network.getFactoryArchitecture());
		ResilientPropagation rprop = new ResilientPropagation(network,trainingSet);
		int iterations=20;
		DecimalFormat df = new DecimalFormat();
		System.out.println("Runing "+iterations+" iterations of training");
		do{
			rprop.iteration();
			System.out.println("Iter # "+rprop.getIteration()+" error "+df.format(rprop.getError()));
		}while(rprop.getIteration()<=iterations);
		rprop.finishTraining();
		
		//Test 100 Random sentences
		Random rnd = new Random();
		int correct=0;
		int incorrect=0;
		for(i=0;i<100;i++){
			int index=rnd.nextInt(sents.size()-1);
			Sentence snt=sents.get(index);
			double testArr[][]={snt.getEncodedText(maxLen)};
			double testLabel[][]={snt.getEncodedLabel()};
			MLDataSet testSet = new BasicNeuralDataSet(testArr, testLabel);
			MLDataPair pair = testSet.get(0);
			final MLData output = network.compute(pair.getInput());
			double networkScore=output.getData(0);
			double realScore=pair.getIdeal().getData(0);
			if(snt.isSentimentPositive()){
				if(networkScore>0.8){
					correct++;
				}else{
					incorrect++;
				}
			}else{
				if(networkScore<=0.8){
					correct++;
				}else{
					incorrect++;
				}
			}
			System.out.println(snt);
			System.out.println("\t Real : "+realScore+" Network : "+df.format(networkScore));
		}
		
		System.out.println("Correct "+correct);
		System.out.println("InCorrect "+incorrect);
	}

}
