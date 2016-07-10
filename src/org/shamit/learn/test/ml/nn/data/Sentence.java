package org.shamit.learn.test.ml.nn.data;

public class Sentence {
	String sentence;
	boolean sentimentPositive;

	public Sentence(String sentence, boolean sentimentPositive) {
		super();
		this.sentence = sentence;
		this.sentimentPositive = sentimentPositive;
	}
	public String getSentence() {
		return sentence;
	}
	public void setSentence(String sentence) {
		this.sentence = sentence;
	}
	public boolean isSentimentPositive() {
		return sentimentPositive;
	}
	public void setSentimentPositive(boolean sentimentPositive) {
		this.sentimentPositive = sentimentPositive;
	}
	
	public double[] getEncodedText(int arrayLen){
		double ret[]=new double[arrayLen];
		for(int i=0;i<sentence.length() && i<arrayLen;i++){
			ret[i]= Character.getNumericValue(sentence.charAt(i));
		}
		return ret;
	}
	
	public double[] getEncodedLabel(){
		double labels[]=new double[1];
		double score=0;
		if(isSentimentPositive()){
			score=1;
		}
		labels[0]=score;
		return labels;
	}
	
	@Override
	public String toString() {
		return "Sentence [sentence=" + sentence + ", sentimentPositive=" + sentimentPositive + "]";
	}
}
