package svmbridge;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.StringTokenizer;

import libsvm.*;

/**
 * A class that provides a simple bridge from svm to Java, without
 * having to re-implement everything from scratch each time.
 * 
 * @author rachelgollub
 *
 */
public class SVMBridge {

  /**
   * Run svm-train on a file of training data.  Writes to a
   * model file called [trainingfile].model  The default params
   * are set to include probability estimates for later precision/
   * recall graphing.
   * 
   * @param trainingfile The name of the training data file.
   */
  public static void trainsvm(String trainingfile, svm_parameter param) {
    try {
      if (param == null) {
        param = setParams();
      }
      svm_problem prob = buildProblem(trainingfile, true);
      svm_model model = svm.svm_train(prob, param);
      svm.svm_save_model(trainingfile + ".model", model);
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  /**
   * Mostly copied from the Java version of svm_predict.  Takes a datafile
   * (with or without labels) and a model file, and writes to
   * [datafile].predict with the prediction results.  Probability estimates
   * are turned on, so the model file has to have been generated with them on.
   * 
   * @param datafile
   * @param modelfile
   */
  public static double predict(String datafile, String modelfile) {
    PrintWriter p = null;
    BufferedReader input = null;
    try {
      int predict_probability = 1;
      svm_model model = svm.svm_load_model(modelfile);
      p = new PrintWriter(new FileWriter(datafile + ".predict"));
      input = new BufferedReader(new FileReader(datafile));

      int correct = 0;
      int total = 0;
      double error = 0;
      double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;

      int svm_type = svm.svm_get_svm_type(model);
      int nr_class = svm.svm_get_nr_class(model);
      double[] prob_estimates = null;

      if (predict_probability == 1) {
        if (svm_type == svm_parameter.EPSILON_SVR
            || svm_type == svm_parameter.NU_SVR) {
          System.out.print("Prob. model for test data: target value = "
              + "predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)"
              + "/(2sigma),sigma=" + svm.svm_get_svr_probability(model) + "\n");
        } else {
          int[] labels = new int[nr_class];
          svm.svm_get_labels(model, labels);
          prob_estimates = new double[nr_class];
          p.print("labels");
          for (int j = 0; j < nr_class; j++) {
            p.print(" " + labels[j]);
          }
          p.print("\n");
        }
      }
      while (true) {
        String line = input.readLine();
        if (line == null)
          break;

        StringTokenizer st = new StringTokenizer(line, " \t\n\r\f:");

        double target = 1.0;
        if (st.countTokens() % 2 == 1) {
          target = Double.parseDouble(st.nextToken());
        }
        int m = st.countTokens() / 2;
        svm_node[] x = new svm_node[m];
        for (int j = 0; j < m; j++) {
          x[j] = new svm_node();
          x[j].index = Integer.parseInt(st.nextToken());
          x[j].value = Double.parseDouble(st.nextToken());
        }

        double v;
        if (predict_probability == 1
            && (svm_type == svm_parameter.C_SVC || svm_type == svm_parameter.NU_SVC)) {
          v = svm.svm_predict_probability(model, x, prob_estimates);
          p.print(v + " ");
          for (int j = 0; j < nr_class; j++)
            p.print(prob_estimates[j] + " ");
          p.print("\n");
        } else {
          v = svm.svm_predict(model, x);
          p.print(v + "\n");
        }

        if (v == target)
          ++correct;
        error += (v - target) * (v - target);
        sumv += v;
        sumy += target;
        sumvv += v * v;
        sumyy += target * target;
        sumvy += v * target;
        ++total;
      }
      if (svm_type == svm_parameter.EPSILON_SVR
          || svm_type == svm_parameter.NU_SVR) {
        System.out.print("Mean squared error = " + error / total
            + " (regression)\n");
        System.out.print("Squared correlation coefficient = "
            + ((total * sumvy - sumv * sumy) * (total * sumvy - sumv * sumy))
            / ((total * sumvv - sumv * sumv) * (total * sumyy - sumy * sumy))
            + " (regression)\n");
      } else {
        System.out.print("Accuracy = " + (double) correct / total * 100 + "% ("
            + correct + "/" + total + ") (classification)\n");
      }
      p.flush();
      p.close();
      input.close();
      return ((double) correct/total * 100.0);
    } catch (Exception e) {
      e.printStackTrace();
      return 0.0;
    } finally {
      try {
        p.close();
      } catch (Exception ex) {
      }
      try {
        input.close();
      } catch (Exception ex) {
      }
    }
  }
  
  /**
   * Just load the model file into memory.
   * 
   * @param modelfile
   * @return
   */
  public static svm_model loadModel(String modelfile) {
    try {
      svm_model model = svm.svm_load_model(modelfile);
      return model;
    } catch (Exception e) {
      System.out.println("Model file " + modelfile + " is missing or malformed.");
      e.printStackTrace();
      return null;
    }
  }
  
  /**
   * Mostly copied from the Java version of svm_predict.  Takes a datafile
   * (with or without labels) and a model file, and writes to
   * [datafile].predict with the prediction results.  Probability estimates
   * are turned on, so the model file has to have been generated with them on.
   * 
   * @param datafile
   * @param modelfile
   */
  public static double inmemPredict(String data, String modelfile) {
    try {
      svm_model model = svm.svm_load_model(modelfile);
      return inmemPredict(data, model);
    } catch (Exception e) {
      System.out.println("Model file " + modelfile + " is missing or malformed.");
      e.printStackTrace();
      return 0.0;
    }
  }
      
  public static double inmemPredict(String data, svm_model model) {
    try {
      int predict_probability = 1;

      int correct = 0;
      int total = 0;
      double error = 0;
      double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;

      int svm_type = svm.svm_get_svm_type(model);
      int nr_class = svm.svm_get_nr_class(model);
      double[] prob_estimates = null;

      if (predict_probability == 1) {
        if (svm_type == svm_parameter.EPSILON_SVR
            || svm_type == svm_parameter.NU_SVR) {
          System.out.print("Prob. model for test data: target value = "
              + "predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)"
              + "/(2sigma),sigma=" + svm.svm_get_svr_probability(model) + "\n");
        } else {
          int[] labels = new int[nr_class];
          svm.svm_get_labels(model, labels);
          prob_estimates = new double[nr_class];
        }
      }
      for (String line : data.split("\\n")) {
        
        StringTokenizer st = new StringTokenizer(line, " \t\n\r\f:");

        double target = 1.0;
        if (st.countTokens() % 2 == 1) {
          target = Double.parseDouble(st.nextToken());
        }
        int m = st.countTokens() / 2;
        svm_node[] x = new svm_node[m];
        for (int j = 0; j < m; j++) {
          x[j] = new svm_node();
          x[j].index = Integer.parseInt(st.nextToken());
          x[j].value = Double.parseDouble(st.nextToken());
        }

        double v;
        if (predict_probability == 1
            && (svm_type == svm_parameter.C_SVC || svm_type == svm_parameter.NU_SVC)) {
          v = svm.svm_predict_probability(model, x, prob_estimates);
          //for (int j = 0; j < nr_class; j++) {
          //  System.out.println("Probability " + j + ": " + prob_estimates[j]);
          //}
        } else {
          v = svm.svm_predict(model, x);
        }

        if (v == target)
          ++correct;
        error += (v - target) * (v - target);
        sumv += v;
        sumy += target;
        sumvv += v * v;
        sumyy += target * target;
        sumvy += v * target;
        ++total;
      }
      if (svm_type == svm_parameter.EPSILON_SVR
          || svm_type == svm_parameter.NU_SVR) {
        System.out.print("Mean squared error = " + error / total
            + " (regression)\n");
        System.out.print("Squared correlation coefficient = "
            + ((total * sumvy - sumv * sumy) * (total * sumvy - sumv * sumy))
            / ((total * sumvv - sumv * sumv) * (total * sumyy - sumy * sumy))
            + " (regression)\n");
      } else {
        System.out.print("Accuracy = " + (double) correct / total * 100 + "% ("
            + correct + "/" + total + ") (classification)\n");
      }
      return ((double) correct/total * 100.0);
    } catch (Exception e) {
      e.printStackTrace();
      return 0.0;
    }
  }

  public static void crossValidate(String trainingfile, svm_parameter param) {
    PrintWriter p = null;
    try {
      if (param == null) {
        param = setParams();
      }
      double nfold = 5.0;
      svm_problem prob = buildProblem(trainingfile, true);
      int leaveout = (int) Math.floor(prob.l / nfold);
      double accuracy = 0.0;
      for (int i=0; i< nfold; i++) {
        p = new PrintWriter(new FileWriter(trainingfile + ".train" + i));
        svm_problem sub = new svm_problem();
        sub.l = prob.l - leaveout;
        sub.y = new double[prob.l - leaveout];
        sub.x = new svm_node[prob.l-leaveout][];
        int index = 0;
        for (int j=0; j<prob.l; j++) {       
          if (j < (i*leaveout) || j >= (i*leaveout + leaveout)) {
            sub.y[index] = prob.y[j];
            sub.x[index] = prob.x[j];
            index++;            
          } else {
            p.print(prob.y[j] + "\t");
            for (svm_node node : prob.x[j]) {
              p.print(" " + node.index + ":" + node.value);
            }
            p.print("\n");
          }
        }
        p.flush();
        p.close();
        svm_model model = svm.svm_train(sub, param);
        svm.svm_save_model(trainingfile + ".model" + i, model);   
        accuracy += predict(trainingfile + ".train" + i, trainingfile + ".model" + i);
      }
      System.out.println("Average accuracy: " + (accuracy/nfold));
    } catch (Exception e) {
      e.printStackTrace();
    }
  }
  /**
   * Generates a precision/recall table from a labeled vectorized
   * classification data file and a predict file (generated by svm-predict or
   * the predict method above).  Writes to [predictfile].pr
   * @param trainingfile
   * @param predictfile
   */
  public static void precisionRecall(String trainingfile, String predictfile,
      double start, double end, double interval) {
    BufferedReader reader = null;
    PrintWriter p = null;
    try {
      svm_problem prob = buildProblem(trainingfile, true);
      reader = new BufferedReader(new FileReader(predictfile));
      String line;
      int lines = 0;
      double[] probability = new double[prob.l];
      while ((line = reader.readLine()) != null) {
        if (!line.startsWith("labels")) {
          String[] vals = line.split("\\s");
          probability[lines] = Double.parseDouble(vals[1]);
          lines++;
        }
      }
      reader.close();
      int num = (int) Math.round((end - start)/interval) + 1;
      double[] precision = new double[num];
      double[] recall = new double[num];
      int ind = 0;
      for (double thresh = start; thresh <= end; thresh += interval) {
        int tp = 0;
        int tn = 0;
        int fp = 0;
        int fn = 0;
        for (int i = 0; i < prob.l; i++) {
          int predicted = (probability[i] <= thresh ? 0 : 1);
          if (predicted == 1 && prob.y[i] == 1) {
            tp++;
          } else if (predicted == 1 && prob.y[i] == 0) {
            fp++;
          } else if (predicted == 0 && prob.y[i] == 1) {
            fn++;
          } else if (predicted == 0 && prob.y[i] == 0) {
            tn++;
          }
        }
        if ((fp + tp) == 0) {
          fp = 1;
        } else if (tp + fn == 0) {
          fn = 1;
        }
	try {
          precision[ind] = (tp * 1.0) / (tp + fp);
          recall[ind] = (tp * 1.0) / (tp + fn);
	} catch (Exception e) {
	}
        ind++;
      }
      p = new PrintWriter(new FileWriter(predictfile + ".pr"));
      for (int i = 0; i < precision.length; i++) {
        p.println(precision[i] + "\t" + recall[i]);
      }
      p.flush();
      p.close();
    } catch (Exception e) {
      e.printStackTrace();
    } finally {
      try {
        reader.close();
      } catch (Exception ex) {
      }
      try {
        p.close();
      } catch (Exception ex) {
      }
    }
  }

  /**
   * Sets the standard parameters: <ul>
   * <li> svm_type = C_SVC
   * <li> kernel_type = LINEAR
   * <li> degree = 3
   * <li> gamma = 0
   * <li> coef0 = 0
   * <li> C = 1
   * <li> nu = .5
   * <li> eps = .001
   * <li> p = .1
   * <li> cache_size = 100
   * <li> probability = 1
   * <li> shrinking = 0
   * <li> nr_weight = 0
   * <li> weight_label = []
   * <li> weight = []
   * </ul>
   * @return
   */
  public static svm_parameter setParams() {
    svm_parameter param = new svm_parameter();
    param.svm_type = svm_parameter.C_SVC;
    param.kernel_type = svm_parameter.LINEAR;
    param.degree = 3;
    param.gamma = 0;
    param.coef0 = 0;
    param.C = 1;
    param.nu = .5;
    param.eps = 0.001;
    param.p = 0.1;
    param.cache_size = 100;
    param.probability = 1;
    param.shrinking = 0;
    param.nr_weight = 0;
    param.weight_label = new int[0];
    param.weight = new double[0];
    return param;
  }

  /**
   * Builds a problem from a vectorized data file.  If the data is pre-labeled,
   * set isTrainingFile to true.
   * @param filename The file to read in.
   * @param isTrainingFile True if the vectors are correctly labeled.
   * @return
   * @throws Exception
   */
  public static svm_problem buildProblem(String filename, boolean isTrainingFile)
      throws Exception {
    BufferedReader reader = new BufferedReader(new FileReader(filename));
    String line;
    svm_problem prob = new svm_problem();
    int lines = 0;
    int features = 0;
    ArrayList<String> lineArray = new ArrayList<String>();
    while ((line = reader.readLine()) != null) {
      lines++;
      lineArray.add(line);
      if (features == 0) {
        features = line.split("\\s+").length - (isTrainingFile ? 1 : 0);
      }
    }
    prob.l = lines;
    prob.y = new double[lines];
    prob.x = new svm_node[lines][];
    int i = 0;
    for (String info : lineArray) {
      String[] entries = info.split("\\s+");
      prob.y[i] = (isTrainingFile ? Double.parseDouble(entries[0]) : 0.0);
      svm_node[] nodes = new svm_node[features];
      int start = 0;
      if (isTrainingFile) {
        start = 1;
      }
      for (int j = start; j < entries.length; j++) {
        String[] vals = entries[j].split(":");
        svm_node node = new svm_node();
        node.index = j;
        node.value = Double.parseDouble(vals[1]);
        nodes[j - 1] = node;
      }
      prob.x[i] = nodes;
      i++;
    }
    return prob;
  }
}
