package org.shamit.learn.test.ml.nn.train;

import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.List;

import org.encog.engine.network.activation.ActivationTANH;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
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
	static MLDataSet testDataSet=null;
	static int maxLen=200;
	static BasicNetwork network = null;
	
	static void loadOrCreateNetwork(String networkFile){
		System.out.println("Using network from file :: "+networkFile);
		File f=new File(networkFile);
		if(f.exists()){
			network=(BasicNetwork)EncogDirectoryPersistence.loadObject(f);
		}else{
			System.out.println("Network file not found, creating it ::  "+networkFile);
			
			network = new BasicNetwork();
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
			
			ResilientPropagation rprop = new ResilientPropagation(network,trainingDataSet);
			int iterations=2000;
			DecimalFormat df = new DecimalFormat();
			System.out.println("Runing "+iterations+" iterations of training");
			do{
				rprop.iteration();
				System.out.println("Iter # "+rprop.getIteration()+" error "+df.format(rprop.getError()));
			}while(rprop.getIteration()<=iterations);
			rprop.finishTraining();
			
			EncogDirectoryPersistence.saveObject(f, network);
			System.out.println("Network saved in file :: "+f.getAbsolutePath());
		}
	}
	
	static MLDataSet createDataSet(List<Sentence> sents){
		System.out.println(sents.size() + " sentences loaded.");
		double data[][]=new double[sents.size()][];
		double labels[][]=new double[sents.size()][];
		int i=0;
		//
		// Initialize data set
		//
		for(Sentence s:sents){
			data[i]=s.getEncodedText(maxLen);
			labels[i]=s.getEncodedLabel();
			i++;
		}
		return new BasicMLDataSet(data, labels);		
	}

	
	static MLDataSet loadOrCreateData(String dataFile, boolean test) throws IOException {
		MLDataSet dataSet=null;
		System.out.println("Using data file :: "+dataFile);
		File f=new File(dataFile);
		if(f.exists()){
			dataSet=EncogUtility.loadEGB2Memory(f);
			System.out.println("Training data loaded from :: "+dataFile);
		}else{
			System.out.println("Data file not found, creating it ::  "+dataFile);
			DataLoader dl = new DataLoader();
			List<Sentence> sents;
			if(test){
				sents = dl.getTestSentences();
			}else{
				sents = dl.getSentences();
			}
			dataSet = createDataSet(sents);
			EncogUtility.saveEGB(f, dataSet);
			System.out.println("Data file created ::  "+dataFile);
		}
		return dataSet;
	}
	
	static void validateBaseDir() {
		File bd = new File(Constants.BASE_DIR);
		if(!bd.exists()){
			System.out.println("Base Directory does not exist :: "+Constants.BASE_DIR);
			Constants.BASE_DIR=new File("."+File.separator).getAbsolutePath();
			System.out.println("Base Directory switched to :: "+Constants.BASE_DIR);
		}
		
	}	
	
	private static void runTest() {
		for(MLDataPair pair:testDataSet){
			final MLData output = network.compute(pair.getInput());
			double networkScore=output.getData(0);
			double realScore=pair.getIdeal().getData(0);
			System.out.println("\t Real : "+realScore+" Network : "+networkScore);

		}
		
	}	
	
	public static void main(String[] args) throws Exception {
		if(args.length>0){
			Constants.BASE_DIR=args[0];
		}
		System.out.println("Using base Directory :: "+Constants.BASE_DIR);
		validateBaseDir();
		String dataFile = Constants.BASE_DIR + File.separator + "traningData.egb";
		String testDataFile = Constants.BASE_DIR + File.separator + "testData.egb";
		File networkSavePath=new File(Constants.BASE_DIR + File.separator + "encog_network.eg");
		trainingDataSet = loadOrCreateData(dataFile,false);
		testDataSet = loadOrCreateData(testDataFile,true); 
		loadOrCreateNetwork(networkSavePath.getAbsolutePath());
		System.out.println("Network structure is :: \n\t"+network.getFactoryArchitecture());
		runTest();
		System.exit(0);
	}




	
	



}
