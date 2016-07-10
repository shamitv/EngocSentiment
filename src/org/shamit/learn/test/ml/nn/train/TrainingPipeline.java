package org.shamit.learn.test.ml.nn.train;

import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.List;

import org.encog.engine.network.activation.ActivationTANH;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.persist.EncogDirectoryPersistence;
import org.encog.util.simple.EncogUtility;
import org.shamit.learn.test.ml.nn.data.DataLoader;
import org.shamit.learn.test.ml.nn.data.Sentence;

public class TrainingPipeline {

	static MLDataSet trainingDataSet=null;
	static int maxLen=200;
	
	static void loadOrCreateData(String dataFile) throws IOException {
		System.out.println("Using data file :: "+dataFile);
		File f=new File(dataFile);
		if(f.exists()){
			trainingDataSet=EncogUtility.loadEGB2Memory(f);
			System.out.println("Training data loaded from :: "+dataFile);
		}else{
			System.out.println("Data file not found, creating it ::  "+dataFile);
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
			
			trainingDataSet = new BasicMLDataSet(data, labels);
			EncogUtility.saveEGB(f, trainingDataSet);
			System.out.println("Data file created ::  "+dataFile);
		}
		
	}
	
	static void validateBaseDir() {
		File bd = new File(Constants.BASE_DIR);
		if(!bd.exists()){
			System.out.println("Base Directory does not exist :: "+Constants.BASE_DIR);
			Constants.BASE_DIR=new File("."+File.separator).getAbsolutePath();
			System.out.println("Base Directory switched to :: "+Constants.BASE_DIR);
		}
		
	}	
	
	public static void main(String[] args) throws Exception {
		if(args.length>0){
			Constants.BASE_DIR=args[0];
		}
		System.out.println("Using base Directory :: "+Constants.BASE_DIR);
		validateBaseDir();
		String dataFile = Constants.BASE_DIR + File.separator + "traningData.egb";
		loadOrCreateData(dataFile);
		BasicNetwork network = new BasicNetwork();
		network.addLayer(new BasicLayer(null,true,maxLen));
		network.addLayer(new BasicLayer(new ActivationTANH(),true,maxLen*2));
		network.addLayer(new BasicLayer(new ActivationTANH(),true,maxLen*2));
		network.addLayer(new BasicLayer(new ActivationTANH(),true,maxLen/2));
		network.addLayer(new BasicLayer(new ActivationTANH(),true,maxLen/(2*2)));
		network.addLayer(new BasicLayer(new ActivationTANH(),true,maxLen/(2*2*2)));
		network.addLayer(new BasicLayer(new ActivationTANH(),true,maxLen/(2*2*2*2)));
		network.addLayer(new BasicLayer(new ActivationTANH(),true,maxLen/(2*2*2*2*2)));
		network.addLayer(new BasicLayer(new ActivationTANH(),false,1));
		network.getStructure().finalizeStructure();
		network.reset();
		
		System.out.println("Network created. Structure is :: \n\t"+network.getFactoryArchitecture());
		ResilientPropagation rprop = new ResilientPropagation(network,trainingDataSet);
		int iterations=2000;
		DecimalFormat df = new DecimalFormat();
		System.out.println("Runing "+iterations+" iterations of training");
		do{
			rprop.iteration();
			System.out.println("Iter # "+rprop.getIteration()+" error "+df.format(rprop.getError()));
		}while(rprop.getIteration()<=iterations);
		rprop.finishTraining();
		File networkSavePath=new File(Constants.BASE_DIR + File.separator + "encog_network.eg");
		EncogDirectoryPersistence.saveObject(networkSavePath, network);
		
	}


	
	



}
