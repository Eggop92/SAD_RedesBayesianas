package Labo3;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.rules.OneR;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;

public class ScanParamsOneR {

	private Instances data;
	private Instances dataR;
	private Instances data_Test;
	private OneR estimador;
	private Evaluation evaluator;
	private int bestK;
	
	public ScanParamsOneR(){}
	
	public static void main(String[] args) throws Exception {
		ScanParamsOneR d = new ScanParamsOneR();
		d.leerArchivo();
		int kMax = d.buscarFuerzaBruta();
		System.out.println("El mejor minBucketSize para los datos es: "+kMax+"");
		//d.cargarArchivosTest();
		d.predecir();
	}
	
	public String pedirDatos(String impr){
		//Imprimimos el mensaje sobre lo que queremos pedir
		System.out.println(impr);
		//Creamos variables y abrimos bufers
		String devolver= "";
		BufferedReader br=null;
		InputStreamReader isr=null;
		try{
			isr= new InputStreamReader(System.in);
			br = new BufferedReader(isr);
		//Obtenemos el dato de consola
			devolver = br.readLine();
		//Cerramos bufers
			br.reset();
			br.close();
			isr.close();
		}catch(IOException e){
		}
			
		return devolver;
	}
	
	public void leerArchivo() throws Exception{
		// pedimos y abrimos el archivo
		String archivo=null;
        FileReader fi=null;
        boolean repetir= true;
        
        while(repetir){
        	archivo = pedirDatos("Introduce el archivo con los datos de entrenamiento");
	        try {
	        	fi= new FileReader(archivo);
	        	repetir= false;
	        } catch (FileNotFoundException e) {
	            System.out.println("ERROR: Revisar path del fichero de datos: "+archivo);
	        }
        }
        // Cargamos las instancias
        data=null;
        try {
            data = new Instances(fi);
        } catch (IOException e) {
            System.out.println("ERROR: Revisar contenido del fichero de datos: "+archivo);
        }
        // Cerramos el archivo
        try {
            fi.close();
        } catch (IOException e) {}
        
        //Randomizamos
        Randomize filter = new Randomize();
        filter.setInputFormat(data);
        dataR = Filter.useFilter(data, filter);
        
        // Seleccionamos el atriburo clase
        dataR.setClassIndex(dataR.numAttributes()-1);
	}
	
	public void cargarArchivosTest() throws Exception{
		// pedimos y abrimos el archivo test
		boolean repetir=true;
		String archivo=null;
		FileReader fi=null;
		while(repetir){
			archivo = pedirDatos("Introduce el archivo con los datos de testeo");
	        try {
	        	fi= new FileReader(archivo);
	        	repetir= false;
	        } catch (FileNotFoundException e) {
	            System.out.println("ERROR: Revisar path del fichero de datos:"+archivo);
	        }
		}
        // Cargamos las instancias
        data_Test=null;
        try {
        	data_Test = new Instances(fi);
        } catch (IOException e) {
            System.out.println("ERROR: Revisar contenido del fichero de datos: "+archivo);
        }
        // Cerramos el archivo
        try {
            fi.close();
        } catch (IOException e) {}
        
        //Determinar la clase
        data_Test.setClassIndex(data_Test.numAttributes()-1);
	}
	
	public int buscarFuerzaBruta() throws Exception{
		//inicializamos las variables
		Evaluation eval;
		double fMeasureMax=0, fM;
		//buscamos los valores posibles
		for (int k=1;k<dataR.numInstances(); k++){
			//entrenamos un caso
			eval=entrenar(k);
			fM=eval.fMeasure(1);
			if(fM>fMeasureMax){
				//Guardamos el mayor en caso de que asi sea
				fMeasureMax=fM;
				bestK=k;
			}
		}
		return bestK;
	}
	
	public Evaluation entrenar(int k) throws Exception{
		//Creamos el estimador
		estimador = new OneR();
		//Seleccionamos parametros
		estimador.setMinBucketSize(k);
		//Creamos y usamos una evaluacion
		 Evaluation evaluator = new Evaluation(dataR);
	     evaluator.crossValidateModel(estimador, dataR, 10, new Random(7));
	     return evaluator;
	}
	
	public void predecir() throws Exception{
		
			//Entrenamos el clasificador
	        evaluator = entrenar(bestK);
	        //Mostramos las figuras de merito
	        System.out.println("Las figuras de merito obtenidas del OneR con los datos introducidos anteriormente son:");
	        System.out.println("Accuracy: "+evaluator.pctCorrect());
			System.out.println("precision: "+evaluator.precision(1));
			System.out.println("f-Measure: "+evaluator.fMeasure(1));
			System.out.println("recall: "+evaluator.recall(1));
			System.out.println("Area under Roc: "+evaluator.areaUnderROC(1));
			//construimos el clasificador
	        estimador.buildClassifier(dataR);
	        
	        //obtenemos las posibles respuestas (las opciones de la clase) y las metemos en un array
	        Enumeration enumerado = dataR.classAttribute().enumerateValues();
	     	ArrayList<String> sols = Collections.list(enumerado);
		     
	     	//Pedimos los datos de prueba
	     	cargarArchivosTest();
	     	
		   //Inicializamos variables que necesitaremos
		    boolean repetir= true;
		    double predictions[] = new double[data_Test.numInstances()];
		    String sFichero = "";
		    File fichero =null;
		    BufferedWriter pw= null;
		     FileWriter fw= null;
		     PrintWriter wr = null;// new PrintWriter(fw); 
		     //Pedimos el archivo donde escribir los resultados
		     while (repetir){
			     sFichero = pedirDatos("Introduce el archivo de destino para los resultados del test. Tiene que finalizar en .txt");
			     if(sFichero.length()>4 && sFichero.substring(sFichero.length()-4, sFichero.length()).equals(".txt")){
			    	 repetir=false;
			     }
		     }
		     fichero = new File(sFichero);
		     try{
		    	 fw= new FileWriter(fichero);
		         pw = new BufferedWriter(fw);
		         wr = new PrintWriter(pw); 
		         wr.write("");
		 //Comenzamos las predicciones y las Guardamos en archivo
		         for (int i = 0; i < data_Test.numInstances(); i++) {
		        	 	predictions[i] = evaluator.evaluateModelOnceAndRecordPrediction(estimador, data_Test.instance(i));
		        	 	wr.append(sols.get((int)predictions[i])+"\n");
		    	 }
		         repetir= false;
		 
		     } catch (Exception e) {
		            e.printStackTrace();
		     }
		    	 wr.close();
		    	 pw.close();
		    	 fw.close();
		}
}
