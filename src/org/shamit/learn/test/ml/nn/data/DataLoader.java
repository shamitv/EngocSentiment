package org.shamit.learn.test.ml.nn.data;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.io.IOUtils;

public class DataLoader {

	List<InputStream> dataStreams = new ArrayList<>();
	String dataFiles[] = {"amazon_cells_labelled.txt","imdb_labelled.txt","yelp_labelled.txt"};
	
	public DataLoader() {
		super();
	}
	
	protected void initStreams(){
		try{
			for(String fname:dataFiles){
				InputStream in=this.getClass().getResourceAsStream(fname);
				if(in==null){
					throw new RuntimeException("Stream not found for file :: "+fname);
				}
				dataStreams.add(in);
			}			
		}catch(Exception e){
			throw new RuntimeException("Error loading training data",e);
		}
	}
	
	List<Sentence> getSentences() throws IOException{
		initStreams();
		List<String> lines=new ArrayList<>();
		List<Sentence> ret=new ArrayList<>();
		for(InputStream in:dataStreams){
			List<String> temp = IOUtils.readLines(in, StandardCharsets.UTF_8);
			lines.addAll(temp);
		}
		for(String line:lines){
			// Format of each line is
			// sentence \t score 
			// First get index of last \t in line 
			int tabIndex=line.lastIndexOf('\t');
			if(tabIndex<1){ 
				//Looks like invalid sentence
				//Skip this one, just print error
				System.err.println("Error in processing line\n"+line);
			}else{
				String text = line.substring(0, tabIndex-1);
				String scoreText = line.substring(tabIndex+1).trim();
				boolean positive=false;
				if(scoreText.trim().equals("1")){
					positive=true;
				}
				Sentence sent = new Sentence(text, positive);
				ret.add(sent);
			}
		}
		return ret;
	}
	
	public static void main(String[] args) throws IOException {
		// Method to unit test loader 
		DataLoader dl = new DataLoader();
		List<Sentence> sents = dl.getSentences();
		int count=0;
		for(Sentence s:sents){
			count++;
			if(count%100==0){
				System.out.println(s);
				System.out.println(Arrays.toString(s.getEncodedText(200)));
			}
		}
	}

}
